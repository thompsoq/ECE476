from tkinter import NW, SW, N, CENTER, Canvas, IntVar, StringVar, SUNKEN, DoubleVar, Toplevel
from tkinter.constants import LEFT, TOP
from tkinter.ttk import Notebook, Frame, Button, Entry, Label, OptionMenu, Checkbutton
from ttkthemes import ThemedTk
from numpy import zeros
from PIL import Image, ImageTk
from cv2 import applyColorMap, convertScaleAbs, COLORMAP_JET
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
from matplotlib.figure import Figure

"""
draw all points or bounding boxes on the canvas 
"""
def draw_points(canvas, actions, point_list, lookup):
    start_point = None
    prev_point = [-1, -1, -1, -1, -1]
    # go through teach of the points
    for i, point in enumerate(point_list):
        # if it is a one coordinate point
        if point[0] == 0:
            actions.append(canvas.object.create_oval(point[1]-5, point[2]-5, point[1]+5, point[2]+5, fill=lookup[i][1]))
        # if it is a multi-coordinate object
        elif point[0] == 1:
            # if the classification is the same as the last
            if point[3] == prev_point[3]:
                # draw a line between the two points
                actions.append(canvas.object.create_line(point[1], point[2], prev_point[1], prev_point[2], width=3, fill = lookup[i][1]))
            # save the first point
            if point[4] == 0:
                start_point = point
            elif point[4] == 2:
                # draw a line between the last and first point
                actions.append(canvas.object.create_line(start_point[1], start_point[2], point[1], point[2], width=3, fill =lookup[i][1]))
            prev_point = point

"""
iterates through the current nodes children in the tkinter object tree
"""
def iterthru_tkinter_objs(search_root, search_name):
    for child in search_root.child:
        if child.nodal_name == search_name:
            return child
    return None

"""
iteratively returns all children from the current node
"""
def iter_return_all(search_root, search_name):
    child_dict = {}
    for child in search_root.child:
        if child.nodal_name.split(':')[0] == search_name:
            child_dict[child.nodal_name.split(':')[1]] = child
    return child_dict

"""
recursively searches list for object and returns it
"""
def recurs_tkinter_objs(search_root, search_name):
    out = None
    for child in search_root.child:
        if child.nodal_name == search_name:
            out = child
            break
        else:
            out = recurs_tkinter_objs(child, search_name)
            if out != None: 
                break
    return out

def recurs_return_all(search_root, insert_dict):
    for child in search_root.child:
        insert_dict[child.nodal_name] = child
        recurs_return_all(child, insert_dict)

def confirmation_popup():
    win = Toplevel()
    win.wm_title("Window")
# why not just have a dict child?
# well, I didn't want to have to define their names both ways for parents and children
# instead, just attaching the name to the node itself
# it leads to an iterative search for finding as opposed to just typing the dict name in
# but it is also less confusing on creation

"""
initializes the object values that all classes share which includes
the parent, setting the parents child, creating the child list and setting the
nodal name
"""
def init_class(node, **kwargs):
    # sets the parent for the current node
    node.parent = kwargs['parent']
    # adds this as a child to the parent node
    node.parent.child.insert(0, node)
    # init the children list for this node
    node.child = []
    # set the name for this node
    node.nodal_name = kwargs['nodal_name']
    # if a header label exists
    if 'header' in kwargs:
        # create the label as a child of this objects parent
        node.header = Label(node.parent.object, text=kwargs['header'])
        # then place it relative to this object
        node.header.place(in_=node.object, bordermode='outside', relx=0.5, rely=0, y=-20, anchor=CENTER)

class Root_ASTF:
    """
    initializes the root window for everything to branch off of
    """
    def __init__(self, parent, **kwargs):
        # does not call init  class because there is no parent, must define child
        self.child = []
        # set the nodal name
        self.nodal_name = kwargs['nodal_name']
        # initialzie the themed Tkinter main window widget
        self.object = ThemedTk()
        # set the default theme to equilux
        self.object.set_theme('equilux')

        # set the geometry to a default window size
        self.object.geometry("1920x1080")
        # full screen the window anyway
        self.object.state('zoomed')

        if 'parent' in kwargs:
            # sets the parent for the current node
            self.parent = kwargs['parent']
            # adds this as a child to the parent node
            self.parent.child.insert(0, self)
            # init the children list for this node
            self.child = []

class Notebook_ASTF:
    """
    initialization of a notebook for tabs
    """
    def __init__(self, parent, **kwargs):
        init_class(node=self, parent=parent, **kwargs)

        self.object = Notebook(parent.object)

class Tab_ASTF:
    """
    initialization of a singular tab for the gui
    """
    def __init__(self, parent, **kwargs):
        init_class(node=self, parent=parent, **kwargs)

        self.tab_name = kwargs['tab_text']
        self.object = Frame(self.parent.object)
        self.parent.object.add(self.object, text=self.tab_name)
        self.parent.object.pack(expand = True, anchor=N, side=LEFT, fill='both', ipadx=5, ipady=3)

class Label_ASTF:
    """
    initalization of a text or image viewable label for the gui
    """
    def __init__(self, parent, **kwargs):
        # if the xml states its a text, create a text box
        if kwargs['type'] == 'text':
            self.create_text_label(parent, kwargs['text'])
        # otherwise create an image label
        if kwargs['type'] == 'image':
            self.create_image_label(parent, kwargs['image_size'])
        # init usually placed after everything else just in case it needs to access this nodes object
        init_class(node=self, parent=parent, **kwargs)


    """
    create a text label and pack it
    """
    def create_text_label(self, parent, text):
        self.object = Label(parent.object, text=text)
        self.object.pack(anchor=N, side=TOP, ipady=28, pady=28)

    """
    create a image label and pack it
    """
    def create_image_label(self, parent, image_size):
        size_width, size_height = image_size.split('x')
        zero_np = zeros((int(size_height), int(size_width)))
        zero_image = Image.fromarray(zero_np)
        self.image = ImageTk.PhotoImage(image=zero_image)
        self.object = Label(parent.object, image=self.image)
        self.object.pack(anchor=N, side=LEFT, ipady=30, pady=30)

class Button_ASTF:
    """
    initializes a button which can be pressed for the gui
    """
    def __init__(self, parent, **kwargs):
        self.object = Button(parent.object, text=kwargs['text'])

        init_class(node=self, parent=parent, **kwargs)

        self.object.pack(anchor=N, side=TOP, ipady=30, pady=30)

class Entry_ASTF:
    """
    initializes an entry field for user input for the gui
    """
    def __init__(self, parent, **kwargs):
        if kwargs['variable'] == 'int':
            self.variable = IntVar(parent.object, int(kwargs['value']))
        elif kwargs['variable'] == 'double':
            self.variable = DoubleVar(parent.object, float(kwargs['value']))
        else:
            self.variable = StringVar(parent.object, kwargs['text'])
        self.object = Entry(parent.object, textvariable=self.variable) 

        init_class(node=self, parent=parent, **kwargs)


        self.object.pack(anchor=N, side=TOP, pady=30)       

class Frame_ASTF:
    """
    initializes a frame to split up the gui
    """
    def __init__(self, parent, **kwargs):
        self.object = Frame(parent.object, relief=SUNKEN)

        init_class(node=self, parent=parent, **kwargs)

        self.object.pack(anchor=N, side=LEFT, fill='both', expand=True, padx=5)

class Frame_Hor_ASTF:
    """
    initializes a frame to split up the gui
    """
    def __init__(self, parent, **kwargs):
        self.object = Frame(parent.object, relief=SUNKEN)

        init_class(node=self, parent=parent, **kwargs)

        self.object.pack(anchor=N, side=TOP, fill='both', expand=True, pady=5)


class Dropdown_ASTF:
    """
    initializes a dropdown window for user selection
    """
    def __init__(self, parent, **kwargs):
        self.variable = StringVar()
        test = kwargs['options'].split(':')
        self.object = OptionMenu(parent.object, self.variable, test[0], *test)

        init_class(node=self, parent=parent, **kwargs)

        self.object.pack(side=TOP, pady=30)

class Canvas_ASTF:
    """
    initializes a canvas for drawing on and viewing images
    """
    def __init__(self, parent, **kwargs):
        self.size_width, self.size_height = kwargs['image_size'].split('x')
        self.object = Canvas(parent.object, bg="black", height=self.size_height, width=self.size_width)
        init_class(node=self, parent=parent, **kwargs)

        self.object.pack(side=TOP, pady=30)
        # image to be drawn too 
        placeholder = zeros((int(self.size_height), int(self.size_width)))
        im = Image.fromarray(placeholder)
        imgtk = ImageTk.PhotoImage(image=im) 
        # create image at center of canvas
        self.image_on_canvas = self.object.create_image(int(self.size_width)//2, int(self.size_height)//2, anchor=CENTER, image = imgtk)

class FigureCanvas_ASTF:
    """
    initializes a canvas for drawing on and viewing images
    """
    def __init__(self, parent, **kwargs):
        self.size_width, self.size_height = kwargs['image_size_inch'].split('x')
        self.dpi = int(kwargs['dpi'])
        self.fig = Figure(figsize = (int(self.size_width), int(self.size_height)),
                   dpi = self.dpi)

        self.fig.patch.set_facecolor((0.275, 0.275, 0.275))

        self.plot = self.fig.add_subplot(111)

        self.plot.xaxis.get_major_locator().set_params(integer=True)
        self.plot.set_xlabel(kwargs['xlabel'])
        self.plot.set_ylabel(kwargs['ylabel'])

        self.plot.set_facecolor((0.275, 0.275, 0.275))

        self.plot.xaxis.label.set_color((0.8, 0.8, 0.8))        #setting up X-axis label color to yellow
        self.plot.yaxis.label.set_color((0.8, 0.8, 0.8))          #setting up Y-axis label color to blue

        self.plot.tick_params(axis='x', colors=(0.8, 0.8, 0.8))    #setting up X-axis tick color to red
        self.plot.tick_params(axis='y', colors=(0.8, 0.8, 0.8))  #setting up Y-axis tick color to black

        self.plot.spines['left'].set_color((0.8, 0.8, 0.8))        # setting up Y-axis tick color to red
        self.plot.spines['top'].set_color((0.8, 0.8, 0.8))         #setting up above X-axis tick color to red


        self.object =  FigureCanvasTkAgg(self.fig, master = parent.object)  
        self.object.draw()
        init_class(node=self, parent=parent, **kwargs)

        self.object.get_tk_widget().pack(side=TOP, pady=30)
        self.toolbar = NavigationToolbar2Tk(self.object,
                                   parent.object)
        self.toolbar.update()

        # placing the toolbar on the Tkinter window
        self.object.get_tk_widget().pack()



class CheckButton_ASTF:
    """
    initializes a checkbutton for true and false input from the user
    """
    def __init__(self, parent, **kwargs):
        self.variable = IntVar()
        self.object = Checkbutton(parent.object, text=kwargs['text'], variable=self.variable)

        init_class(node=self, parent=parent, **kwargs)
        self.object.pack(side=TOP, pady=30)
