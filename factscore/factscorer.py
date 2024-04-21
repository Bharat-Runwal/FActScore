import argparse
import string
import json
import numpy as np
import os
import logging
import time 
from collections import defaultdict

from tqdm import tqdm
from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.clm import CLM
from factscore.npm import NPM
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval

class FactScorer(object):

    def __init__(self,
                 model_name="retrieval+ChatGPT",
                 data_dir=".cache/factscore",
                 model_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 openai_org=None,
                 cost_estimate="consider_cache",
                 abstain_detection_type=None,
                 batch_size=256,
                 af_model_name="ChatGPT",
                 af_model_version=None,
                 verbose=False):
        assert model_name in ["retrieval+llama", "retrieval+llama+npm", "retrieval+ChatGPT", "npm", "retrieval+ChatGPT+npm"]
        self.model_name = model_name
        self.verbose = verbose
        if self.verbose:
            print(f"Model name = {model_name}")

        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size # batch size for retrieval
        self.openai_key = openai_key
        self.openai_org = openai_org
        self.af_model_name = af_model_name
        self.af_model_version = af_model_version
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.cost_estimate = cost_estimate

        if "llama" in model_name:
            self.lm = CLM("inst-llama-7B",
                          model_dir=os.path.join(model_dir, "inst-llama-7B"),
                          cache_file=os.path.join(cache_dir, "inst-llama-7B.pkl"))
        elif "ChatGPT" in model_name:
            self.lm = OpenAIModel("ChatGPT",
                                  cache_file=os.path.join(cache_dir, "ChatGPT.pkl"),
                                  key_path=openai_key)
        else:
            self.lm = None

    def save_cache(self):
        if self.lm:
            self.lm.save_cache()
        if "npm" in self.model_name:
            for k, v in self.npm.items():
                v.save_cache()
        for k, v in self.retrieval.items():
            v.save_cache()

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)
        if "npm" in self.model_name:
            cache_path = os.path.join(self.cache_dir, f"bm25-{name}.json")
            embed_cache_path = os.path.join(self.cache_dir, f"bm25-{name}.pkl")
            self.npm[name] = NPM(Retrieval(self.db[name], cache_path, embed_cache_path, "bm25"),
                                 "npm-single",
                                 data_dir = self.data_dir,
                                 cache_file=os.path.join(self.cache_dir, f"npm-{name}.pkl"))


    def print_cost_estimates(self, total_words, task, model):
        # https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
        # Number of tokens are roughly 4/3 of the number of words
        total_tokens = total_words * 4.0 / 3

        # https://openai.com/pricing
        # if we use davinci-003, the cost is $0.02 per 1000 tokens
        # if we use gpt-3.5-turbo, the cost is $0.002 per 1000 tokens
        if model == "davinci-003":
            rate = 0.02
        elif model == "gpt-3.5-turbo":
            rate = 0.0015
        elif model == "gpt-3.5-turbo-instruct":
            rate = 0.002
        elif model == "gpt-4":
            rate = 0.06

        total_cost = total_tokens * rate / 1000

        # print the total words, tokens, and cost along with rate
        logging.critical("Estimated OpenAI API cost for %s ($%.3f per 1000 tokens): $%.3f for %d words and %d tokens" % (task, rate, total_cost, total_words, total_tokens))

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False,
                  chunk_size=0,
                  af_model_name=None,# or "InstructGPT"
                  is_bio=True,
                  num_ex=4,
                  ):

        self.verbose = verbose 
        start_time = time.time()

        # register knowledge source 
        if knowledge_source is None:
            # use the default one (enwiki-20230401)
            knowledge_source = "enwiki-20230401"
            if knowledge_source not in self.retrieval:
                self.register_knowledge_source(knowledge_source)
        else:
            assert knowledge_source in self.retrieval, \
                f"{knowledge_source} is not registered yet. Please use `register_knowledge_source()` function to register it with a database"
        
        if type(topics)==type(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        # atomic facts (added af_model_name option)
        if af_model_name:
            self.af_model_name = af_model_name
        print(f"af_model_name: {self.af_model_name}")
        logging.critical(f"af_model_name: {self.af_model_name}")
        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key, 
                                                        is_bio=is_bio,
                                                        num_ex=num_ex,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, f"{self.af_model_name}.pkl"), 
                                                        af_model_name=self.af_model_name,
                                                        af_model_version=self.af_model_version,
                                                        openai_org=self.openai_org)

            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            if self.af_model_name == "ChatGPT":
                openai_model_name = "gpt-3.5-turbo"
            elif self.af_model_name == "InstructGPT":
                openai_model_name = "gpt-3.5-turbo-instruct"
            else:
                raise NotImplementedError(f"af_model_name {af_model_name} is not supported")
            
            self.print_cost_estimates(total_words, 
                                      task=f"atomic fact generation w/ {af_model_name} [topic {topics[0]}]", 
                                      model=openai_model_name)
          
            if verbose:
                topics = tqdm(topics)

            atomic_start_time = time.time()
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 100 == 0:
                    # cache_start_time = time.time()
                    self.af_generator.save_cache()
                    # print(f"Save AF cache took {time.time() - cache_start_time} seconds")
            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()

            fact_gen_time = time.time() - atomic_start_time
            print(f"Atomic fact gen time: {fact_gen_time:.4f}")

        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if self.verbose:
            topics = tqdm(topics)

        out_list = []
        total_time_dict = {"fact_gen_time": fact_gen_time, "retrieval_time": 0, "generation_time": 0, "npm_time": 0}
        if self.verbose:
            print("LEN TOPICS:", len(topics))
        # scores_list = []

        for i, (topic, generation, facts) in enumerate(zip(topics, generations, atomic_facts)):
            scores = []
            init_scores = []
            decisions = []
            if facts is None:
                decisions.append(None)
            else:
                if chunk_size == 0:
                    decision, time_dict = self._get_score(topic, generation, facts, knowledge_source)
                else:
                    decision, time_dict = self._get_score_batched(topic, generation, facts, knowledge_source, chunk_size=chunk_size)
                score = np.mean([d["is_supported"] for d in decision])
                total_time_dict["retrieval_time"] += time_dict["retrieval_time"]
                total_time_dict["generation_time"] += time_dict["generation_time"]
                total_time_dict["npm_time"] += time_dict["npm_time"]
                if self.verbose:
                    print(f"TIME DICT {i}:", time_dict)

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                # print("DECISION:", decision)
                # print("SCORE:", score)
            # print(f"KAT: obtained scores for {topic}")
            
            # if i % 35 == 0:
            #     # print("save cache")
            #     cache_start_time = time.time()
            #     self.save_cache()
            #     print(f"KAT: save cache took {time.time() - cache_start_time} seconds")

            # print("LEN DECISIONS:", len(decisions))
            # print("LEN SCORES:", len(scores))
            out = {
                "topic": topic,
                "generation": generation,
                "score": np.mean(scores),
                "respond_ratio": respond_ratio,
                "decisions": decisions,
                "num_facts_per_response": np.mean([len(d[0]) for d in decisions if d is not None])
            }
            if gamma:
                out["init_score"] = np.mean(init_scores)
            out_list.append(out)
        self.save_cache()
        if True: # self.verbose 
            print("total time dict:", total_time_dict)
        return out_list
    
    # this *_batch function aims to generate a batch of generations for speedup
    # not tested/used currently 
    '''
    def get_score_batched(self,
                  topics,
                  generations,
                  gamma=10,
                  atomic_facts=None,
                  knowledge_source=None,
                  verbose=False,
                  chunk_size=8):
        
        start_time = time.time()

        if knowledge_source is None:
            # use the default one (enwiki-20230401)
            knowledge_source = "enwiki-20230401"
            if knowledge_source not in self.retrieval:
                self.register_knowledge_source(knowledge_source)
        else:
            assert knowledge_source in self.retrieval, \
                f"{knowledge_source} is not registered yet. Please use `register_knowledge_source()` function to register it with a database"
        # print(f"KAT: registered knowledge source at time {time.time() - start_time:.4f} seconds")
        if type(topics)==len(generations)==str:
            topics = [topics]
            generations = [generations]
        else:
            assert type(topics)==type(generations)==list, "`topics` and `generations` should be lists."
            assert len(topics)==len(generations), "`topics` and `generations` should have the same length"

        if atomic_facts is not None:
            assert len(topics)==len(atomic_facts), "`topics` and `atomic_facts` should have the same length"
        else:
            af_load_start_time = time.time()
            if self.af_generator is None:
                self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                        demon_dir=os.path.join(self.data_dir, "demos"),
                                                        gpt3_cache_file=os.path.join(self.cache_dir, "InstructGPT.pkl"))
            fact_gen_time = time.time() - af_load_start_time
            print(f"KAT: load af generator time: {time.time() - af_load_start_time:.4f}")
            # estimate the total cost of atomic fact generation
            total_words = 0
            for gen in generations:
                total_words += self.af_generator.run(gen, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="atomic fact generation", model="davinci-003")

            if verbose:
                topics = tqdm(topics)

            atomic_start_time = time.time()
            atomic_facts = []
            for topic, gen in zip(topics, generations):
                # optionally, first detect if the response is abstained
                response_abstained = is_response_abstained(gen, self.abstain_detection_type)
                if response_abstained:
                    atomic_facts.append(None)
                    continue
                # continue only when the response is not abstained
                curr_afs, _ = self.af_generator.run(gen)
                curr_afs = [fact for _, facts in curr_afs for fact in facts]
                if len(curr_afs)==0:
                    atomic_facts.append(None)
                else:
                    atomic_facts.append(curr_afs)
                if len(atomic_facts) % 50 == 0:
                    # cache_start_time = time.time()
                    self.af_generator.save_cache()
                    # print(f"KAT: save cache took {time.time() - cache_start_time} seconds")

            assert len(atomic_facts)==len(topics)
            self.af_generator.save_cache()
            print(f"KAT: obtained atomic facts with time {time.time() - atomic_start_time:.4f}")
            if verbose:
                print("Atomic facts:", atomic_facts)


        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if "ChatGPT" in self.model_name:
            # estimate the total cost of response generation
            total_words = 0
            for topic, generation, facts in zip(topics, generations, atomic_facts):
                if facts is not None:
                    total_words += self._get_score(topic, generation, facts, knowledge_source, cost_estimate=self.cost_estimate)

            self.print_cost_estimates(total_words, task="factscore evaluation", model="gpt-3.5-turbo")

        if verbose:
            topics = tqdm(topics)

        out_list = []
        total_time_dict = {"num_topics": len(topics), "fact_gen_time": fact_gen_time, "retrieval_time": 0, "generation_time": 0, "npm_time": 0}
        # print("LEN TOPICS:", len(topics))
        # scores_list = []

        def group_same_elements(input_list):
            element_indices = {}
            # grouped_indices = []
            for idx, element in enumerate(input_list):
                if element in element_indices:
                    element_indices[element].append(idx)
                else:
                    element_indices[element] = [idx]
            # for indices in element_indices.values():
            #     grouped_indices.append(indices)
            return element_indices
        
        topic_inds = group_same_elements(topics)
        fact_dict = {}
        for topic in topic_inds.keys():
            fact_dict[topic] = defaultdict(list)
        for i, (topic, generation, facts) in enumerate(zip(topics, generations, atomic_facts)):
            print(i, topic, facts)
            fact_dict[topic]["facts"].extend(facts)
            fact_dict[topic]["inds"].extend([i] * len(facts))

        for topic in fact_dict.keys():
            decision, time_dict = self._get_score_batched(topic, "", fact_dict[topic]["facts"], knowledge_source, chunk_size=8)
            print("len facts", len(fact_dict[topic]["facts"]))
            print("len decision", len(decision))
            

        for i, (topic, generation, facts) in enumerate(zip(topics, generations, atomic_facts)):
            scores = []
            init_scores = []
            decisions = []
            if facts is None:
                decisions.append(None)
            else:
                if chunk_size == 1:
                    decision, time_dict = self._get_score(topic, generation, facts, knowledge_source)
                else:
                    decision, time_dict = self._get_score_batched(topic, generation, facts, knowledge_source, chunk_size=chunk_size)
                score = np.mean([d["is_supported"] for d in decision])
                total_time_dict["retrieval_time"] += time_dict["retrieval_time"]
                total_time_dict["generation_time"] += time_dict["generation_time"]
                total_time_dict["npm_time"] += time_dict["npm_time"]
                print(f"TIME DICT {i}:", time_dict)

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts)>gamma else np.exp(1-gamma/len(facts))
                    score = penalty * score
                
                decisions.append(decision)
                scores.append(score)
                # print("DECISION:", decision)
                # print("SCORE:", score)
            # print(f"KAT: obtained scores for {topic}")
            
            # if i % 35 == 0:
            #     # print("save cache")
            #     cache_start_time = time.time()
            #     self.save_cache()
            #     print(f"KAT: save cache took {time.time() - cache_start_time} seconds")

            # print("LEN DECISIONS:", len(decisions))
            # print("LEN SCORES:", len(scores))
            out = {
                "topic": topic,
                "generation": generation,
                "score": np.mean(scores),
                "respond_ratio": respond_ratio,
                "decisions": decisions,
                "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])
            }
            if gamma:
                out["init_score"] = np.mean(init_scores)
            out_list.append(out)
        self.save_cache()
        print("total time dict:", total_time_dict)
        return out_list
    '''
    
    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        # start_time = time.time()
        decisions = []
        total_words = 0

        retrieval_total_time = 0
        generation_total_time = 0
        npm_total_time = 0
        for atom in atomic_facts:
            atom = atom.strip()
            if self.lm:
                retrieval_time_start = time.time()
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
                retrieval_time = time.time() - retrieval_time_start

                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue
                    
                generation_time_start = time.time()
                output = self.lm.generate(prompt)
                generation_time = time.time() - generation_time_start
                
                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            else:
                is_supported = True

            npm_time_start = time.time()
            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3
            npm_time = time.time() - npm_time_start

            retrieval_total_time += retrieval_time
            generation_total_time += generation_time
            npm_total_time += npm_time

            decisions.append({"atom": atom, "is_supported": is_supported})
        # print(f"_get_score took {time.time() - start_time} seconds")
        if cost_estimate:
            return total_words
        else:
            return decisions, {"retrieval_time": retrieval_total_time, "generation_time": generation_total_time, "npm_time": npm_total_time}
    
    '''
    def _get_score_batched(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None, chunk_size = 8):
        if self.verbose:
            print(f"_get_score_batched for {len(atomic_facts)} atomic facts")
        start_time = time.time()
        decisions = []
        total_words = 0
        prompt_list = []
        if self.lm:
            retrieval_start_time = time.time()
            for i, atom in enumerate(atomic_facts):
                atom = atom.strip()
                passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5) # this line produces a TQDM of 1 batch each time
                # print("got passages")
                definition = "Answer the question about {} based on the given context.\n\n".format(topic)
                context = ""
                for psg_idx, psg in enumerate(reversed(passages)):
                    context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
                definition += context.strip()
                if not definition[-1] in string.punctuation:
                    definition += "."
                prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
                prompt_list.append(prompt)
                if cost_estimate:
                    if cost_estimate == "consider_cache" and (prompt.strip() + "_0") not in self.lm.cache_dict:
                        total_words += len(prompt.split())
                    elif cost_estimate == "ignore_cache":
                        total_words += len(prompt.split())
                    continue
                    
            # print("prompt list:", len(prompt_list))
            # output_list = self.lm.generate_batch(prompt_list)
            retrieval_time = time.time() - retrieval_start_time
            print("retrieval time", retrieval_time)
            generation_start_time = time.time()
            output_list = self.lm.generate_batch(prompt_list, chunk_size=chunk_size)
            # output_list = [] 
            # for i in list(range(0, len(prompt_list), chunk_size)):
            #     output_list.extend(self.lm.generate_batch(prompt_list[i:i+chunk_size]))
            generation_time = time.time() - generation_start_time
            # print("output list", len(output_list), output_list)
            # print(f"generated all outputs by {time.time() - start_time} seconds")
            supported_list = []
            for i, output in enumerate(output_list):
                if type(output[1])==np.ndarray:
                    # when logits are available
                    logits = np.array(output[1])
                    assert logits.shape[0] in [32000, 32001]
                    true_score = logits[5852]
                    false_score = logits[7700]
                    is_supported = true_score > false_score
                else:
                    # when logits are unavailable
                    generated_answer = output[0].lower()
                    if "true" in generated_answer or "false" in generated_answer:
                        if "true" in generated_answer and "false" not in generated_answer:
                            is_supported = True
                        elif "false" in generated_answer and "true" not in generated_answer:
                            is_supported = False
                        else:
                            is_supported = generated_answer.index("true") > generated_answer.index("false")
                    else:
                        is_supported = all([keyword not in generated_answer.lower().translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])
                supported_list.append(is_supported)
        else:
            supported_list = [True for i in range(len(atomic_facts))]
            
        npm_time_start = time.time()
        for i, atom in enumerate(atomic_facts):
            is_supported = supported_list[i]
            if is_supported and "npm" in self.model_name:
                npprob = self.npm[knowledge_source].get_probabilty(topic, atom)
                is_supported = npprob > 0.3
            decisions.append({"atom": atom, "is_supported": is_supported})
        npm_time = time.time() - npm_time_start

        time_dict = {"retrieval_time": retrieval_time, "generation_time": generation_time, "npm_time": npm_time}
        # print(f"retrieval_time: {retrieval_time}, generation_time: {generation_time}, npm_time: {npm_time}")

        # print(f"_get_score_batched took {time.time() - start_time} seconds")
        if cost_estimate:
            return total_words
        else:
            return decisions, time_dict
    '''

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path',
                        type=str,
                        default="data/labeled/InstructGPT.jsonl")
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+ChatGPT")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--model_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")    
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    tot = 0
    topics, generations, atomic_facts = [], [], []
    with open(args.input_path) as f:
        for line in f:
            dp = json.loads(line)
            tot += 1
            if args.use_atomic_facts:
                assert "annotations" in dp, "You can specify `--use_atomic_facts` only when atomic facts are available in the input data already."
                if dp["annotations"] is None:
                    continue
                topics.append(dp["topic"])
                generations.append(dp["output"])
                atomic_facts.append([atom["text"] for sent in dp["annotations"] for atom in sent["model-atomic-facts"]])
            else:
                topics.append(dp["topic"])
                generations.append(dp["output"])
            if args.n_samples is not None and tot==args.n_samples:
                break
    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       verbose=args.verbose)
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))



