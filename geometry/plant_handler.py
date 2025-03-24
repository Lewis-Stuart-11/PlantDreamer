import os 

#from abc import ABC, abstractmethod

from .l_systems import get_l_system

class PlantType():
    def __init__(self, plant_species):
        self.plant_species = plant_species
        self.set_prompt()
        self.set_lora_name()
        self.l_system_path = None

    # The prompt to use for each plant in the diffusion process
    def set_prompt(self):
        self.prompt = f"A photo of {self.plant_species} plants" 

    # The name of the file path to the saved LoRAs
    def set_lora_name(self):
        self.lora_name = os.path.join(self.plant_species, "pytorch_lora_weights.safetensors")

    # Method that will be overwritten that generates an L-System mesh
    def get_l_systems_mesh(self):
        self.l_system_path = get_l_system(self.plant_species) 

class Bean(PlantType):
    def __init__(self):
        super().__init__("bean")

class Kale(PlantType):
    def __init__(self):
        super().__init__("kale") 

class Mint(PlantType):
    def __init__(self):
        super().__init__("mint")

""" 
    Add your Species subclass here
"""

# Factory that returns the plant object for the plant species string
def get_plant_type(plant_type_str):
    plant_type_str = plant_type_str.lower().strip()

    """ 
        Add your string to species condiution here
    """ 
    if plant_type_str == "bean":
        return Bean()
    elif plant_type_str == "kale":
        return Kale()
    elif plant_type_str == "mint":
        return Mint()
    else:
        raise Exception("Plant Species not supported- if this is a new species, add this to the Plant_Handler")

    