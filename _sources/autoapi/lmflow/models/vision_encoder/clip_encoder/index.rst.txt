:py:mod:`lmflow.models.vision_encoder.clip_encoder`
===================================================

.. py:module:: lmflow.models.vision_encoder.clip_encoder


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   lmflow.models.vision_encoder.clip_encoder.CLIPVisionTower



Functions
~~~~~~~~~

.. autoapisummary::

   lmflow.models.vision_encoder.clip_encoder.build_vision_tower



.. py:function:: build_vision_tower(vision_tower_cfg, **kwargs)


.. py:class:: CLIPVisionTower(vision_tower, args, delay_load=False)

   Bases: :py:obj:`torch.nn.Module`

   
   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool















   ..
       !! processed by numpydoc !!
   .. py:property:: dummy_feature


   .. py:property:: dtype


   .. py:property:: device


   .. py:property:: config


   .. py:property:: hidden_size


   .. py:property:: num_patches


   .. py:method:: load_model()


   .. py:method:: encode_images(images, language_projection)


   .. py:method:: feature_select(image_forward_outs)


   .. py:method:: forward(images)


   .. py:method:: prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, language_projection=None, language_model=None, **kwargs)

      
      Copy from the LLAVA code base.
      Should be polished.
















      ..
          !! processed by numpydoc !!


