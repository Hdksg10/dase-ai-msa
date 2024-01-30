from transformers import BertTokenizerFast, ViTImageProcessor
from transformers import ProcessorMixin
from transformers import BatchEncoding

import torch

class MultiModalProcessor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "ViTImageProcessor"
    tokenizer_class = ("BertTokenizer", "BertTokenizerFast")
    
    def __init__(self):
        image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        super().__init__(image_processor=image_processor, tokenizer=tokenizer)
        
    def __call__(self, text=None, images=None, return_tensors=None, **kwargs):
        if text is None and images is None:
            raise ValueError("You have to specify either text or images. Both cannot be none.")
        
        if text is not None:
            encoding = self.tokenizer(text, return_tensors=return_tensors, **kwargs)

        if images is not None:
            image_features = self.image_processor(images, return_tensors=return_tensors, **kwargs)
    
        if text is not None and images is not None:
            encoding["pixel_values"] = image_features.pixel_values
            return encoding
        elif text is not None:
            return encoding
        else:
            return BatchEncoding(data=dict(**image_features), tensor_type=return_tensors)
        
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
    

if __name__ == '__main__':
    from sadatasets import TrainDataset, TestDataset
    processor = MultiModalProcessor()
    train_dataset = TrainDataset('../datasets')
    test_dataset = TestDataset('../datasets')
    for data in train_dataset:
        print(data)
        output = processor(data['text'], data['image'], padding=True, truncation=True, return_tensors="pt")
        print(output)
        break