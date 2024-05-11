import sys
import time
import configparser
import xml.etree.ElementTree as ET
from configparser import NoSectionError, NoOptionError

import pandas as pd
import numpy as np


class XMLHandler:
    def __init__(self, filepath):
        self.filepath = filepath
        self.tree = ET.ElementTree()
        self.root = None
        self.load_xml()

    def load_xml(self):
        try:
            self.tree = ET.parse(self.filepath)
            self.root = self.tree.getroot()
        except ET.ParseError:
            print(f"XML file {self.filepath} not found or parse fail！")

    def print_xml_content(self):
        
        def recursive_print(element,indent=""):
            print(f"{indent} Tag: {element.tag}, Attribute: {element.attrib}")
            if element.text:
                print(f"{indent} Text: {element.text.strip()}")
            for child in element:
                recursive_print(child,indent + ' ')
        
        recursive_print(self.root)
        
    def save_xml(self, output_filepath):
        try:
            self.tree.write(output_filepath, encoding="utf-8", xml_declaration=True)
            print("XML file save succeed！")
        except IOError:
            print("XML file save fail！")

    def find_elements(self, xpath, display=False):
        self.load_xml()
        elements = self.root.findall(xpath)
        if elements:
            if display:
                for element in elements:
                    element_str = ET.tostring(element,encoding="utf-8").decode("utf-8")
            return elements
        else:
            print(f"elements not found for {xpath}！")
            return []

    def display_element_with_attribute(self,attribute,value):
        target_node = None
        for node in self.root.iter():
            if attribute in node.attrib and node.attrib['name'] == value:
                target_node = node
                break
        if target_node is not None:
            print(target_node.attrib)
        else:
            print(f"elements not found for {attribute}！")

    def add_element_with_attribute(self,attribute,value,add_attribute,add_value):
        target_node = None
        for node in self.root.iter():
            if attribute in node.attrib and node.attrib['name'] == value:
                target_node = node
                break
        if target_node is not None:
            target_node.set(add_attribute,add_value)
            print("attribute has been set!")
        else:
            print(f'node not found for {attribute}!')
        self.save_xml(output_filepath=self.filepath)

    def add_element_with_custom_name(self,tag_name,old_name,new_name):
        for element in self.root.findall(tag_name):
            if element.get('name')  == old_name:
                new_element = ET.Element(tag_name)
                for attr, value in element.items():
                    if attr == 'name':
                        value = new_name
                    new_element.set(attr, value)
                self.root.append(new_element)
        self.save_xml(output_filepath=self.filepath)

    def remove_attribute_from_xml(self, attribute_name):
        for element in self.root.iter():
            if attribute_name in element.attrib:
                del element.attrib[attribute_name]
        self.save_xml(output_filepath=self.filepath)

    def remove_element_by_name(self,tag_name,name):
        for element in self.root.findall(tag_name):
            if element.get('name') == name:
                self.root.remove(element)
        self.save_xml(output_filepath=self.filepath)

        
def Read_parser(lcfFile, ConfigFile, SimParamsFile):
    ### Read Layer File
    thickness_layers = {}
    try:
        lcf_df = pd.read_csv(lcfFile, lineterminator="\n")
        if(lcf_df['Thickness (m)'].isnull().values.any()):
            print('Error:', 'Thickness (m) must be specified for each layer'); sys.exit(2)
        thickness_layers = lcf_df.set_index("Layer").to_dict()["Thickness (m)"]
        if(lcf_df['FloorplanFile'].isnull().values.any()):
            print('Error:', 'Floorplan File must be specified for each layer'); sys.exit(2)
    except FileNotFoundError:
        print('Error:', 'Layer File does not exist:', lcfFile); sys.exit(2)

    ###### Read Default config file and SimParams file; default format is ordered dictionary
    defaultConfig = configparser.ConfigParser()
    try:
        defaultConfig.read(ConfigFile)
    except FileNotFoundError:
        print('Error:', 'Config File does not exist:', ConfigFile); sys.exit(2)

    SimParams = configparser.ConfigParser()
    try:
        SimParams.read(SimParamsFile)
    except FileNotFoundError:
        print('Error:', 'SimParams File does not exist:', SimParamsFile); sys.exit(2)

    ###### Add Package Layer
    num_layers = lcf_df['Layer'].max()
    if "NoPackage" in SimParams:
        noPackage_layer = pd.DataFrame([], columns=[
            'Layer','Main_compo','Thickness (m)','FloorplanFile','PtraceFile','Clip_num_x','Clip_num_y','Clip_num_z'])
        noPackage_layer.loc[0, 'Layer'] = num_layers+1
        noPackage_layer.loc[0, 'Main_compo'] = "NoPackage"
        noPackage_layer.loc[0, 'FloorplanFile'] = lcf_df.loc[num_layers, 'FloorplanFile']
        noPackage_layer.loc[0, 'PowerFile'] = None
        noPackage_layer.loc[0, 'Thickness (m)'] = defaultConfig.get('NoPackage', 'thickness (m)')
        noPackage_layer.loc[0, 'Clip_num_x'] = defaultConfig.get('NoPackage', 'clip_num_x')
        noPackage_layer.loc[0, 'Clip_num_y'] = defaultConfig.get('NoPackage', 'clip_num_y')
        noPackage_layer.loc[0, 'Clip_num_z'] = defaultConfig.get('NoPackage', 'clip_num_z')
        lcf_df = pd.concat([lcf_df, noPackage_layer], ignore_index=True, sort=False)
        thickness_layers[num_layers+1] = float(defaultConfig.get('NoPackage', 'thickness (m)'))
        
    ### Read all unique floorplan files names
    ### Create tuples of config_file and floorplan_file
    ### Check if the details of the  materials to be modeled are present
    flp_files = lcf_df['FloorplanFile'].unique()
    config_label_df = pd.DataFrame()
    for ff in flp_files:
        try:
            ff_df = pd.read_csv(ff)
        except FileNotFoundError:
            print('Error: Floorplan file not found', ff); sys.exit(2)

        config_label_df = pd.concat([config_label_df, 
                                 ff_df[['Label']].drop_duplicates()], ignore_index=True, sort=False)
    list_of_labels = config_label_df['Label'].unique()
    if "NoPackage" in SimParams:
        list_of_labels = np.append(list_of_labels, ['NoPackage'], axis=0)
    label_dict = dict.fromkeys([(x) for x in list_of_labels])

    ### Ordered dictionary with all material properties from the config file
    label_properties_dict = {}
    for ll in list_of_labels:
        try:
            lib_location = SimParams.get(ll, 'library')
        except NoOptionError:
            print('ERROR: Library not defined for the label \'', ll, '\''); sys.exit(2)
        try:
            lib_name = SimParams.get(ll, 'library_name')
        except NoOptionError:
            print('ERROR: Library_name not defined for the label \'', ll, '\''); sys.exit(2)
        try:
            properties = [option for option in defaultConfig[ll]]
            properties_needed = [x.strip() for x in SimParams.get(lib_name, 'properties').split(',')]
        except (NoSectionError):
            print('ERROR: Label \'', ll, '\' not defined in the config file.'); sys.exit(2)
    
    return thickness_layers, lcf_df, defaultConfig, SimParams, label_dict