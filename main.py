from src import DatasetGenerator
import xml.etree.ElementTree as ET

tree = ET.parse('config.xml')
xmlRoot = tree.getroot()

def createds():
    DatasetGenerator.generateDataset(xmlRoot.find('datasetPath').text, xmlRoot.find('datasetBckPath').text)

createds()