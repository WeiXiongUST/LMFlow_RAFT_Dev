:py:mod:`lmflow.utils.llava_conversation_lib`
=============================================

.. py:module:: lmflow.utils.llava_conversation_lib


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.utils.llava_conversation_lib.SeparatorStyle
   lmflow.utils.llava_conversation_lib.Conversation




Attributes
~~~~~~~~~~

.. autoapisummary::

   lmflow.utils.llava_conversation_lib.conv_vicuna_v0
   lmflow.utils.llava_conversation_lib.conv_vicuna_v1
   lmflow.utils.llava_conversation_lib.conv_llama_2
   lmflow.utils.llava_conversation_lib.conv_llava_llama_2
   lmflow.utils.llava_conversation_lib.conv_mpt
   lmflow.utils.llava_conversation_lib.conv_llava_plain
   lmflow.utils.llava_conversation_lib.conv_llava_v0
   lmflow.utils.llava_conversation_lib.conv_llava_v0_mmtag
   lmflow.utils.llava_conversation_lib.conv_llava_v1
   lmflow.utils.llava_conversation_lib.conv_llava_v1_mmtag
   lmflow.utils.llava_conversation_lib.default_conversation
   lmflow.utils.llava_conversation_lib.conv_templates


.. py:class:: SeparatorStyle(*args, **kwds)

   Bases: :py:obj:`enum.Enum`

   
   Different separator style.
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: SINGLE
      

      

   .. py:attribute:: TWO
      

      

   .. py:attribute:: MPT
      

      

   .. py:attribute:: PLAIN
      

      

   .. py:attribute:: LLAMA_2
      

      


.. py:class:: Conversation

   
   A class that keeps all conversation history.
















   ..
       !! processed by numpydoc !!
   .. py:attribute:: system
      :annotation: :str

      

   .. py:attribute:: roles
      :annotation: :List[str]

      

   .. py:attribute:: messages
      :annotation: :List[List[str]]

      

   .. py:attribute:: offset
      :annotation: :int

      

   .. py:attribute:: sep_style
      :annotation: :SeparatorStyle

      

   .. py:attribute:: sep
      :annotation: :str = ###

      

   .. py:attribute:: sep2
      :annotation: :str

      

   .. py:attribute:: version
      :annotation: :str = Unknown

      

   .. py:attribute:: skip_next
      :annotation: :bool = False

      

   .. py:method:: get_prompt()


   .. py:method:: append_message(role, message)


   .. py:method:: get_images(return_pil=False)


   .. py:method:: to_gradio_chatbot()


   .. py:method:: copy()


   .. py:method:: dict()



.. py:data:: conv_vicuna_v0
   

   

.. py:data:: conv_vicuna_v1
   

   

.. py:data:: conv_llama_2
   

   

.. py:data:: conv_llava_llama_2
   

   

.. py:data:: conv_mpt
   

   

.. py:data:: conv_llava_plain
   

   

.. py:data:: conv_llava_v0
   

   

.. py:data:: conv_llava_v0_mmtag
   

   

.. py:data:: conv_llava_v1
   

   

.. py:data:: conv_llava_v1_mmtag
   

   

.. py:data:: default_conversation
   

   

.. py:data:: conv_templates
   

   

