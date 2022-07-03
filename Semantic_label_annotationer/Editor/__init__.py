'''
design mode: simple factory
'''
import importlib
from .base_editor import BaseEditor
def find_editor_class_by_name(name):#ex:name="Merge"
    cls_name = name+'Editor'
    file_name = "Editor.{}_editor".format(name.lower())
    module = importlib.import_module(file_name)
    assert cls_name in module.__dict__, 'Cannot find Edior name "{}" in "{}"'.format(cls_name,file_name)
    cls = module.__dict__[cls_name]
    assert issubclass(cls,BaseEditor) , 'Editor class "{}" must inherit from BaseEditor'.format(cls_name)

    return cls
def create_editor(name):
    editor = find_editor_class_by_name(name)
    print('Creating "{}" Editor...'.format(name))
    instance = editor()
    return instance
