import pickle
import os
import time

class LM(object):

    def __init__(self, cache_file, read_only_cache=None):
        self.cache_file = cache_file
        self.read_only_cache = read_only_cache
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    # this *_batch function aims to generate a batch of generations for speedup
    # not tested/used currently 
    def generate_batch(self, prompt_batch, sample_idx=0, max_sequence_length=2048, max_output_length=128, chunk_size=8):
        prompt_batch = [prompt.strip() for prompt in prompt_batch] # it's important not to end with a whitespace
        cache_key_batch = [f"{prompt}_{sample_idx}" for prompt in prompt_batch]
        
        generated_batch = {}
        inference_batch = []
        inference_inds = []
        for i, cache_key in enumerate(cache_key_batch):
            if cache_key in self.cache_dict:
                generated_batch[i] = self.cache_dict[cache_key]
                # print("Cache hit:", i, "-", generated_batch[i])
            else:
                inference_batch.append(prompt_batch[i])
                inference_inds.append(i)
        
        print(f"{len(generated_batch)} out of {len(prompt_batch)} are already cached. Need to run inference for items {inference_inds}.")
        if len(inference_batch) == 0:
            return [generated_batch[i] for i in range(len(prompt_batch))]
        

        model_load_start = time.time()
        if self.model is None:
            self.load_model()
        model_load_time = time.time() - model_load_start
        print(f"Model load time: {model_load_time:.4f}")

        generated_texts_list = []
        generated_arrs_list = []
        ends_with_list = [prompt.endswith(" True or False?\nAnswer:") for prompt in inference_batch]
        for i in list(range(0, len(inference_batch), chunk_size)):
            ends_with = ends_with_list[i:i+chunk_size]
            # output_list.extend(self.lm.generate_batch(inference_batch[i:i+chunk_size]))
            if sum(ends_with) == len(ends_with):
                generated = self._generate_batch(inference_batch, max_sequence_length=max_sequence_length, max_output_length=1)
                generated_texts, generated_arrs = generated
            elif sum(ends_with) == 0:
                generated = self._generate_batch(inference_batch, max_sequence_length=max_sequence_length, max_output_length=max_output_length)
                generated_texts, generated_arrs = generated 
            else:
                print("ERROR: Mixed prompts in batch, generating separately...")
            generated_texts_list.extend(generated_texts)
            generated_arrs_list.extend(generated_arrs)

        # print("Generated:",len(generated), generated)
        # print("Inference inds:", inference_inds)
        # print("Cache keys:", cache_key_batch)

        for i, ind in enumerate(inference_inds):
            generated_batch[ind] = (generated_texts[i], generated_arrs[i])
            self.cache_dict[cache_key_batch[ind]] = (generated_texts[i], generated_arrs[i])

        generated_batch_list = [generated_batch[i] for i in range(len(prompt_batch))]
        self.add_n += len(inference_batch)
        return generated_batch_list

    def save_cache(self):
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)
        print("(Lm) finished saving cache")

    def load_cache(self, allow_retry=True):
        start_time = time.time()
        print("(Lm) Loading cache from", self.cache_file)
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        print("starting to try loading file")
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print("Pickle Error: Retry in 5 sec...")
                    time.sleep(5)
        else:
            cache = {}
        
        # if self.read_only_cache and os.path.exists(self.read_only_cache):
        #     while True:
        #         try:
        #             with open(self.read_only_cache, "rb") as f:
        #                 ro_cache = pickle.load(f)
        #             break
        #         except Exception:
        #             if not allow_retry:
        #                 assert False
        #             print ("Pickle Error: Retry in 3 sec...")
        #             time.sleep(3)
        #     for k,v in ro_cache.items():
        #         if k not in cache:
        #             cache[k] = v

        print(f"(Lm) Finished {len(cache)} items in {time.time() - start_time:.3f} sec.")
        return cache



