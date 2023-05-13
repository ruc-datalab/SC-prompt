from typing import List
from torch import nn
import torch
from transformers.modeling_utils import PreTrainedModel
import numpy as np


class PromptWrapper(nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        prompt_length_list: list = [60, 15, 15, 60],
        initial_vectors_path: str = '',
        freeze_model: bool = True,
        stage: str = 'content',
        initialize_from_pretrain: bool = False,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True
    ):
        super().__init__()

        self.prompt_length = sum(prompt_length_list)
        self.model = model
        if freeze_model == True:
            for p in model.parameters():
                p.requires_grad = False
        if not initialize_from_pretrain:
            self.prompt_head = nn.Parameter(
                self.initialize_embedding(
                    model.get_input_embeddings(),
                    prompt_length_list[0],
                    random_range,
                    initialize_from_vocab,
                )
            )
            self.prompt_mid1 = nn.Parameter(
                self.initialize_embedding(
                    model.get_input_embeddings(),
                    prompt_length_list[1],
                    random_range,
                    initialize_from_vocab,
                )
            )
            self.prompt_mid2 = nn.Parameter(
                self.initialize_embedding(
                    model.get_input_embeddings(),
                    prompt_length_list[2],
                    random_range,
                    initialize_from_vocab,
                )
            )
            self.prompt_tail = nn.Parameter(
                self.initialize_embedding(
                    model.get_input_embeddings(),
                    prompt_length_list[3],
                    random_range,
                    initialize_from_vocab,
                )
            )
        else:
            print(f"initialize from {initial_vectors_path}")

            self.prompt_head = nn.Parameter(
                torch.from_numpy(np.load(initial_vectors_path))
            )    
            self.prompt_tail = nn.Parameter(
                torch.from_numpy(np.load(initial_vectors_path.replace('head', 'tail')))
            )
            self.prompt_mid1 = nn.Parameter(
                torch.from_numpy(np.load(initial_vectors_path.replace('head', 'mid1')))
            )
            self.prompt_mid2 = nn.Parameter(
                torch.from_numpy(np.load(initial_vectors_path.replace('head', 'mid2')))
            ) 
        

    def initialize_embedding(
        self,
        embedding: nn.Embedding,
        prompt_length: int = 10,
        random_range: float = 0.5,
        initialize_from_vocab: bool = True,
        initialize_from_keywords: bool = True,
    ):

        if initialize_from_vocab:     
            indices = torch.randint(0, 5000, (prompt_length,))
            return embedding.weight[indices].clone().detach()
        
        return torch.FloatTensor(prompt_length, embedding.weight.size(1)).uniform_(
            -random_range, random_range
        )

    def build_inputs(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        device = input_ids.device

        prompt_length = self.prompt_head.size(0) + self.prompt_mid1.size(0) + self.prompt_mid2.size(0) + self.prompt_tail.size(0)
        if prompt_length and attention_mask is not None:
            padding = torch.full((batch_size, (prompt_length)), 1).to(device)
            attention_mask = torch.cat((padding, attention_mask), dim=1)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        """
        Input Format:
        [prompt_head] Translate the question into sql according to the database: xxx 
        [prompt_mid1] | question: xxx 
        [prompt_mid2] | database: xxx 
        [prompt_tail]
        """
        extend_embeds = []
        for idx0 in range(inputs_embeds.shape[0]):
            mid1 = None
            mid2 = None
            end = None
            for idx1 in range(inputs_embeds.shape[1]):
                if input_ids[idx0][idx1] == 1820 and input_ids[idx0][idx1+1] == 822 and mid1 == None:
                    mid1 = idx1
                elif input_ids[idx0][idx1] == 1820 and input_ids[idx0][idx1+1] == 3501 and mid2 == None:
                    mid2 = idx1
                elif input_ids[idx0][idx1] == 1:
                    end = idx1
            if mid2 == None:
                mid2 = end

            extend_embeds.append(torch.cat([self.prompt_head, inputs_embeds[idx0][:mid1], self.prompt_mid1, inputs_embeds[idx0][mid1:mid2], self.prompt_mid2, inputs_embeds[idx0][mid2:end], self.prompt_tail, inputs_embeds[idx0][end:]], 0))


        inputs_embeds = torch.stack(extend_embeds,dim=0)

        return inputs_embeds, attention_mask, labels

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        inputs_embeds, attention_mask, labels = self.build_inputs(
            input_ids,
            attention_mask,
            labels,
        )

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    @torch.no_grad()
    def generate(self, input_ids=None, attention_mask=None, structures=None, **kwargs):
        inputs_embeds, attention_mask, _ = self.build_inputs(
            input_ids,
            attention_mask,
            labels=None,
        )

        model_kwargs = {
            "encoder_outputs": self.model.get_encoder()(inputs_embeds=inputs_embeds)
        }

        return self.model.generate(
            input_ids=None,
            use_cache=True,
            no_repeat_ngram_size=0,
            structures=structures,
            **model_kwargs,
            **kwargs,
        )

    @property
    def config(self):
        return self.model.config
