from abc import ABC, abstractmethod

"""Abstract class, methods for cleaning input and output to models"""

class Processor(ABC):
    @abstractmethod
    def preprocess_function(self, input_feature):
        """Defines function for processing inputs"""
    
    @abstractmethod
    def postprocess_function(self, output_feature):
        """Defines function for processing outputs"""