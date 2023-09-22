:py:mod:`lmflow.models.vision2seq_model`
========================================

.. py:module:: lmflow.models.vision2seq_model


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.models.vision2seq_model.CustomAutoVision2SeqModel




.. py:class:: CustomAutoVision2SeqModel(config: transformers.Blip2Config, image_encoder_name_or_path=None, qformer_name_or_path=None, language_model_name_or_path=None, low_resource=False)

   Bases: :py:obj:`transformers.Blip2ForConditionalGeneration`, :py:obj:`lmflow.models.base_model.BaseModel`

   
   An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
   models.
















   ..
       !! processed by numpydoc !!
   .. py:method:: get_backend_model()


   .. py:method:: vision_model_from_pretrained(pretrained_path)


   .. py:method:: qformer_from_pretrained(pretrained_path)


   .. py:method:: language_model_from_pretrained(pretrained_path, low_resource=False, use_prompt_cache=False)


   .. py:method:: vision_feature_select(image_forward_outs)


   .. py:method:: register_prompt_cache(prompt_ids, prompt_keys_values)

      
      Udpate the prompt id and embedding for reuse in the future

      Args:
          prompt_ids (torch.LongTensor): The id of the prompt.
          prompt_keys_values (torch.FloatTensor): The embedding of the prompt.

      Returns:
          None















      ..
          !! processed by numpydoc !!

   .. py:method:: save_prompt_cache(path)

      
      Save prompt embedding and id.

      Args:
          path: The path to save the prompt embedding and id.

      Returns:
          None















      ..
          !! processed by numpydoc !!

   .. py:method:: load_prompt_cache(path)

      
      Load prompt embedding and id.
      Args:
          path: The path to load the prompt embedding and id.

      Returns:
          None















      ..
          !! processed by numpydoc !!

   .. py:method:: get_tokenizer()


   .. py:method:: forward(input_ids: torch.LongTensor = None, pixel_values: Optional[torch.FloatTensor] = None, images: Optional[torch.FloatTensor] = None, attention_mask: Optional[torch.Tensor] = None, past_key_values: Optional[List[torch.FloatTensor]] = None, inputs_embeds: Optional[torch.FloatTensor] = None, labels: Optional[torch.LongTensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None, image_token_indexes: Optional[List] = [0], one_sample_multiple_images: bool = False) -> Union[Tuple, transformers.modeling_outputs.CausalLMOutputWithPast]

      
      Returns:

      Examples:

      Image captioning (without providing a text prompt):

      ```python
      >>> from PIL import Image
      >>> import requests
      >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
      >>> import torch

      >>> device = "cuda" if torch.cuda.is_available() else "cpu"

      >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
      >>> model = Blip2ForConditionalGeneration.from_pretrained(
      ...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
      ... )
      >>> model.to(device)  # doctest: +IGNORE_RESULT

      >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
      >>> image = Image.open(requests.get(url, stream=True).raw)

      >>> inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

      >>> generated_ids = model.generate(**inputs)
      >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
      >>> print(generated_text)
      two cats laying on a couch
      ```

      Visual question answering (prompt = question):

      ```python
      >>> from PIL import Image
      >>> import requests
      >>> from transformers import Blip2Processor, Blip2ForConditionalGeneration
      >>> import torch

      >>> device = "cuda" if torch.cuda.is_available() else "cpu"

      >>> processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
      >>> model = Blip2ForConditionalGeneration.from_pretrained(
      ...     "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
      ... )
      >>> model.to(device)  # doctest: +IGNORE_RESULT

      >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
      >>> image = Image.open(requests.get(url, stream=True).raw)

      >>> prompt = "Question: how many cats are there? Answer:"
      >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

      >>> generated_ids = model.generate(**inputs)
      >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
      >>> print(generated_text)
      two
      ```















      ..
          !! processed by numpydoc !!

   .. py:method:: processor_image_token_in_minigpt4(input_ids, language_model_inputs, attention_mask, image_token_indexes, pixel_values, batch_size=1)


   .. py:method:: generate(pixel_values: torch.FloatTensor, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.LongTensor] = None, image_token_indexes: Optional[List] = [0], one_sample_multiple_images: Optional[bool] = False, images: Optional[torch.LongTensor] = None, **generate_kwargs) -> torch.LongTensor

      
      Overrides `generate` function to be able to use the model as a conditional generator.

      Args:
          pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
              Input images to be processed.
          input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
              The sequence used as a prompt for the generation.
          attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
              Mask to avoid performing attention on padding token indices
          image_token_indexes (bool, *optional*):
              The index for inserting the image tokens.
          one_sample_multiple_images: (bool, *optional*):
              The flag for inference that the input batch size is 1 and contain multiple images.

      Returns:
          captions (list): A list of strings of length batch_size * num_captions.















      ..
          !! processed by numpydoc !!


