import xml.etree.ElementTree as ET
import os
from network_files.pytorch_layers import Network
from network_files.pytorch_test import Test_Network
from code_tkinter.tkinter_objects import Canvas_ASTF, CheckButton_ASTF, Dropdown_ASTF, FigureCanvas_ASTF, Frame_Hor_ASTF, Notebook_ASTF, Tab_ASTF, Label_ASTF, Root_ASTF, Button_ASTF, Entry_ASTF, Frame_ASTF, recurs_tkinter_objs


tkinter_obj_lookup = {
    'Root': Root_ASTF,
    'Notebook': Notebook_ASTF,
    'Tab': Tab_ASTF,
    'Label': Label_ASTF,
    'Button': Button_ASTF,
    'Entry': Entry_ASTF,
    'Frame': Frame_ASTF,
    'Frame_Hor': Frame_Hor_ASTF,
    'Dropdown': Dropdown_ASTF,
    'Canvas': Canvas_ASTF,
    'FigureCanvas': FigureCanvas_ASTF,
    'CheckButton': CheckButton_ASTF
}

"""
parse the XML tree and attribute each of the xml entrees
while attributing them to an object 
"""
def tree_parse(xml_node, gui_parent):
    # recursively populate tkinter widgets to gui
    xml_fields = {}
    for xml_child in xml_node:
        if xml_child.tag in tkinter_obj_lookup:
            tree_parse (xml_child, tkinter_node)
        else:
            xml_fields[xml_child.tag] = xml_child.text
            if xml_child.tag == 'nodal_name':
                tkinter_node = tkinter_obj_lookup[xml_node.tag](gui_parent, **xml_fields)


    return tkinter_node

"""
Main statement to start the program and initialize the different objects
"""

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    xml_tree = ET.parse("code_tkinter/tkinter_obj.xml")
    xml_tree_root = xml_tree.getroot()


    root = tree_parse(xml_tree_root, None)


    network = Network(recurs_tkinter_objs(root, 'tab_net'))

    test_net = Test_Network(recurs_tkinter_objs(root, 'tab_test'))

    root.object.mainloop()
