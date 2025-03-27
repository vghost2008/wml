# Copyright (c) OpenMMLab. All rights reserved.
import ast
import copy
import os
import os.path as osp
import platform
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from argparse import Action, ArgumentParser
from collections import abc
from importlib import import_module,invalidate_caches
import json
from addict import Dict
from yapf.yapflib.yapf_api import FormatCode
import yaml
import wml.wml_utils as wmlu
from .misc import import_modules_from_strings
from .path import check_file_exist

if platform.system() == 'Windows':
    import regex as re
else:
    import re

BASE_KEY = '_base_'
DELETE_KEY = '_delete_'
OVERRIDE_KEY = "_override_"
DEPRECATION_KEY = '_deprecation_'
MERGE_KEY = "_merge_"
RESERVED_KEYS = ['filename', 'text', 'pretty_text']


class ConfigDict(Dict):

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(f"'{self.__class__.__name__}' object has no "
                                f"attribute '{name}'")
        except Exception as e:
            ex = e
        else:
            return value
        raise ex
     
    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            >>> # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = self
        cfg_dict = ConfigDict._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys)
        self.update(cfg_dict)

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """
        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = ConfigDict._merge_a_into_b(v, b[k], allow_list_keys)
            elif isinstance(v, dict):
                if v.pop(OVERRIDE_KEY,False) and k not in b:
                    info = f"ERROR: {OVERRIDE_KEY}==True and key {k} not in base."
                    raise RuntimeError(info)
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = ConfigDict._merge_a_into_b(v, b[k], allow_list_keys)
                else:
                    b[k] = ConfigDict(v)
            else:
                b[k] = v
        return b


def add_args(parser, cfg, prefix=''):
    for k, v in cfg.items():
        if isinstance(v, str):
            parser.add_argument('--' + prefix + k)
        elif isinstance(v, int):
            parser.add_argument('--' + prefix + k, type=int)
        elif isinstance(v, float):
            parser.add_argument('--' + prefix + k, type=float)
        elif isinstance(v, bool):
            parser.add_argument('--' + prefix + k, action='store_true')
        elif isinstance(v, dict):
            add_args(parser, v, prefix + k + '.')
        elif isinstance(v, abc.Iterable):
            parser.add_argument('--' + prefix + k, type=type(v[0]), nargs='+')
        else:
            print(f'cannot parse key {prefix + k} of type {type(v)}')
    return parser


class Config:
    """A facility for config and config files.

    It supports common file formats as configs: python/json/yaml. The interface
    is the same as a dict object and also allows access config values as
    attributes.

    Example:
        >>> cfg = Config(dict(a=1, b=dict(b1=[0, 1])))
        >>> cfg.a
        1
        >>> cfg.b
        {'b1': [0, 1]}
        >>> cfg.b.b1
        [0, 1]
        >>> cfg = Config.fromfile('tests/data/config/a.py')
        >>> cfg.filename
        "/home/kchen/projects/mmcv/tests/data/config/a.py"
        >>> cfg.item4
        'test'
        >>> cfg
        "Config [path: /home/kchen/projects/mmcv/tests/data/config/a.py]: "
        "{'item1': [1, 2], 'item2': {'a': 0}, 'item3': True, 'item4': 'test'}"
    """

    ph_regexp_pre = r'\bvars\(?\)?.'
    ph_regexp = ph_regexp_pre+'([\w_]+)'
    @staticmethod
    def _validate_py_syntax(filename,ph_values={}):
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            content = f.read()
        try:
            if ph_values is not None and len(ph_values)>0:
                content = Config._remove_ph_wrap(content,ph_values=ph_values)
            ast.parse(content)
        except SyntaxError as e:
            raise SyntaxError('There are syntax errors in config '
                              f'file {filename}: {e}')

    @staticmethod
    def _substitute_predefined_vars(filename, temp_config_name,ph_values={}):
        '''
        处理预定义的值，如fileDirname, fileBasename, fileExtname等
        '''
        filename = osp.abspath(osp.expanduser(filename))
        file_dirname = osp.dirname(filename)
        dir_basename = osp.basename(file_dirname)
        file_basename = osp.basename(filename)
        file_basename_no_extension = osp.splitext(file_basename)[0]
        file_extname = osp.splitext(filename)[1]
        support_templates = dict(
            fileDirname=file_dirname,
            dirBasename = dir_basename,
            fileBasename=file_basename,
            fileBasenameNoExtension=file_basename_no_extension,
            fileExtname=file_extname)
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        for key, value in support_templates.items():
            regexp = r'\{\{\s*' + str(key) + r'\s*\}\}'
            value = value.replace('\\', '/')
            config_file = re.sub(regexp, value, config_file) #将regexp替换为value
        if ph_values is not None and len(ph_values)>0:
            config_file = Config._remove_ph_wrap(config_file,ph_values=ph_values)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)

    @staticmethod
    def _pre_substitute_base_vars(filename, temp_config_name):
        """Substitute base variable placehoders to string, so that parsing
        would work."""
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            config_file = f.read()
        base_var_dict = {}
        regexp = r'\{\{\s*' + BASE_KEY + r'\.([\w\.]+)\s*\}\}'
        base_vars = set(re.findall(regexp, config_file))
        for base_var in base_vars:
            randstr = f'_{base_var}_{uuid.uuid4().hex.lower()[:6]}'
            base_var_dict[randstr] = base_var
            regexp = r'\{\{\s*' + BASE_KEY + r'\.' + base_var + r'\s*\}\}'
            config_file = re.sub(regexp, f'"{randstr}"', config_file)
        with open(temp_config_name, 'w', encoding='utf-8') as tmp_config_file:
            tmp_config_file.write(config_file)
        return base_var_dict

    @staticmethod
    def _substitute_base_vars(cfg, base_var_dict, base_cfg):
        """Substitute variable strings to their actual values."""
        '''
        将base文件中的值合并到当前文件中
        示例:

        dict(
        type='Expand',
        mean={{_base_.model.data_preprocessor.mean}},
        to_rgb={{_base_.model.data_preprocessor.bgr_to_rgb}},
        ratio_range=(1, 4)),
        '''
        cfg = copy.deepcopy(cfg)

        if isinstance(cfg, dict):
            for k, v in cfg.items():
                if isinstance(v, str) and v in base_var_dict:
                    new_v = base_cfg
                    for new_k in base_var_dict[v].split('.'):
                        new_v = new_v[new_k]
                    cfg[k] = new_v
                elif isinstance(v, (list, tuple, dict)):
                    cfg[k] = Config._substitute_base_vars(
                        v, base_var_dict, base_cfg)
        elif isinstance(cfg, tuple):
            cfg = tuple(
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg)
        elif isinstance(cfg, list):
            cfg = [
                Config._substitute_base_vars(c, base_var_dict, base_cfg)
                for c in cfg
            ]
        elif isinstance(cfg, str) and cfg in base_var_dict:
            new_v = base_cfg
            for new_k in base_var_dict[cfg].split('.'):
                new_v = new_v[new_k]
            cfg = new_v

        return cfg

    @staticmethod
    def _file2dict(filename, use_predefined_variables=True,ph_values={}):
        filename = osp.abspath(osp.expanduser(filename))
        check_file_exist(filename)
        fileExtname = osp.splitext(filename)[1]
        if fileExtname not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')

        with tempfile.TemporaryDirectory() as temp_config_dir:
            temp_config_file = tempfile.NamedTemporaryFile(
                dir=temp_config_dir, suffix=fileExtname)
            if platform.system() == 'Windows':
                temp_config_file.close()
            temp_config_name = osp.basename(temp_config_file.name)
            # Substitute predefined variables
            if use_predefined_variables:
                Config._substitute_predefined_vars(filename,
                                                   temp_config_file.name,
                                                   ph_values=ph_values)
            else:
                shutil.copyfile(filename, temp_config_file.name)
            # Substitute base variables from placeholders to strings
            base_var_dict = Config._pre_substitute_base_vars(
                temp_config_file.name, temp_config_file.name)

            if filename.endswith('.py'):
                temp_module_name = osp.splitext(temp_config_name)[0]
                sys.path.insert(0, temp_config_dir)
                Config._validate_py_syntax(filename,ph_values=ph_values)
                mod = import_module(temp_module_name)
                sys.path.pop(0)
                cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                # delete imported module
                del sys.modules[temp_module_name]
            elif filename.endswith('.json'):
                with open(temp_config_file.name,"r") as f:
                    cfg_dict = json.load(f)
            elif filename.endswith(('.yml', '.yaml')):
                with open(temp_config_file.name,"r") as f:
                    cfg_dict = yaml.load(f)

            # close temp file
            temp_config_file.close()

        # check deprecation information
        if DEPRECATION_KEY in cfg_dict:
            deprecation_info = cfg_dict.pop(DEPRECATION_KEY)
            warning_msg = f'The config file {filename} will be deprecated ' \
                'in the future.'
            if 'expected' in deprecation_info:
                warning_msg += f' Please use {deprecation_info["expected"]} ' \
                    'instead.'
            if 'reference' in deprecation_info:
                warning_msg += ' More information can be found at ' \
                    f'{deprecation_info["reference"]}'
            warnings.warn(warning_msg, DeprecationWarning)

        cfg_text = filename + '\n'
        with open(filename, 'r', encoding='utf-8') as f:
            # Setting encoding explicitly to resolve coding issue on windows
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:
            cfg_dir = osp.dirname(filename)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = base_filename if isinstance(
                base_filename, list) else [base_filename]

            cfg_dict_list = list()
            cfg_text_list = list()
            for f in base_filename:
                _cfg_dict, _cfg_text = Config._file2dict(osp.join(cfg_dir, f),ph_values=ph_values)
                cfg_dict_list.append(_cfg_dict)
                cfg_text_list.append(_cfg_text)

            base_cfg_dict = dict()
            for i,c in enumerate(cfg_dict_list):
                duplicate_keys = base_cfg_dict.keys() & c.keys()
                if len(duplicate_keys) > 0:
                    '''
                    引入文件中有重复的关键字，需要合并引入文件中的关键字
                    '''
                    #raise KeyError('Duplicate key is not allowed among bases. '
                                   #f'Duplicate keys: {duplicate_keys}')
                    for key in duplicate_keys:
                        if c[key] is None:
                            #如果当前base文件中的值为None,使用其它base文件中的值
                            print(f"WARNING: Find duplicate key {key} in config, value in {base_filename[i]} is None, use old value {base_cfg_dict[key]}")
                            c.pop(key)
                        else:
                            #有重复且都不为空，使用后引入的base文件的值
                            print(f"WARNING: Find duplicate key {key} in config, use {base_filename[i]} for key {key}")
                        '''elif isinstance(c[key],dict) and isinstance(base_cfg_dict[key],dict):
                            new_value = c[key]
                            new_value.update(base_cfg_dict[key])
                            print(f"WARNING: Find duplicate key {key} in config, value in {base_filename[i]} is:\n {c[key]} \n, old value is:\n {base_cfg_dict[key]}\n merged to:\n {new_value}")
                            c[key] = new_value
                        else:
                            print(f"WARNING: Find duplicate key {key} in config, use {base_filename[i]} for key {key}")
                        '''
                    base_cfg_dict.update(c)
                else:
                    base_cfg_dict.update(c)

            # Substitute base variables from strings to their actual values
            cfg_dict = Config._substitute_base_vars(cfg_dict, base_var_dict,
                                                    base_cfg_dict)

            base_cfg_dict = Config._merge_a_into_b(cfg_dict, base_cfg_dict)
            cfg_dict = base_cfg_dict

            # merge cfg_text
            cfg_text_list.append(cfg_text)
            cfg_text = '\n'.join(cfg_text_list)


        return cfg_dict, cfg_text

    @staticmethod
    def _merge_a_into_b(a, b, allow_list_keys=False,is_overide=False):
        """merge dict ``a`` into dict ``b`` (non-inplace).

        Values in ``a`` will overwrite ``b``. ``b`` is copied first to avoid
        in-place modifications.

        Args:
            a (dict): The source dict to be merged into ``b``.
            b (dict): The origin dict to be fetch keys from ``a``.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in source ``a`` and will replace the element of the
              corresponding index in b if b is a list. Default: False.

        Returns:
            dict: The modified dict of ``b`` using ``a``.

        Examples:
            # Normally merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # Delete b first and merge a into b.
            >>> Config._merge_a_into_b(
            ...     dict(obj=dict(_delete_=True, a=2)), dict(obj=dict(a=1)))
            {'obj': {'a': 2}}

            # b is a list
            >>> Config._merge_a_into_b(
            ...     {'0': dict(a=2)}, [dict(a=1), dict(b=2)], True)
            [{'a': 2}, {'b': 2}]
        """

        if isinstance(b,(int,float,str,bytes)):
            return a

        b = b.copy()
        for k, v in a.items():
            if allow_list_keys and k.isdigit() and isinstance(b, list):
                k = int(k)
                if len(b) <= k:
                    raise KeyError(f'Index {k} exceeds the length of list {b}')
                b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys,is_overide=is_overide)
            elif isinstance(v, dict):
                n_is_overide = v.pop(OVERRIDE_KEY,is_overide)
                if n_is_overide and k not in b:
                    info = f"ERROR: {OVERRIDE_KEY}==True and key {k} not in base."
                    raise RuntimeError(info)
                if k in b and not v.pop(DELETE_KEY, False):
                    allowed_types = (dict, list) if allow_list_keys else dict
                    if b[k] is None:
                        b[k] = v
                    elif not isinstance(b[k], allowed_types):
                        raise TypeError(
                            f'{k}={v} in child config cannot inherit from '
                            f'base because {k} is a dict in the child config '
                            f'but is of type {type(b[k])} in base config. '
                            f'You may set `{DELETE_KEY}=True` to ignore the '
                            f'base config.')
                    b[k] = Config._merge_a_into_b(v, b[k], allow_list_keys,is_overide=n_is_overide)
                else:
                    b[k] = ConfigDict(v)
            elif isinstance(v,(list,tuple)) and k in b and isinstance(b[k],(list,tuple)):
                #合并list, tuple
                if MERGE_KEY not in v:
                    b[k] = v
                else:
                    v = list(v)
                    idx = v.index(MERGE_KEY)
                    del v[idx]

                    b_v = b[k]
                    merge_size = min(len(v),len(b[k]))
                    new_v = []
                    for i in range(merge_size):
                        if isinstance(v[i],(dict,list,tuple)):
                            cur_v = Config._merge_a_into_b(v[i],b_v[i])
                        else:
                            cur_v = v[i]
                        new_v.append(cur_v)
                    new_v.extend(list(v[merge_size:]))
                    new_v.extend(list(b_v[merge_size:]))
                    b[k] = new_v
            else:
                b[k] = v
        return b

    @staticmethod
    def fromfile(filename,
                 use_predefined_variables=True,
                 import_custom_modules=True):
        ph_values = Config.get_placeholder_values(filename)
        if len(ph_values)>0:
            print(f"Placeholder values:")
            for k,v in ph_values.items():
                print(f"{k:<20}: {v}")
        cfg_dict, cfg_text = Config._file2dict(filename,
                                               use_predefined_variables,
                                               ph_values=ph_values)
        if import_custom_modules and cfg_dict.get('custom_imports', None):
            import_modules_from_strings(**cfg_dict['custom_imports'])
        cfg_dict = Config.remove_tmp_value(cfg_dict)
        cfg = Config(cfg_dict, cfg_text=cfg_text, filename=filename)
        Config.set_sibling_key_vals(cfg)
        Config.set_key_vals(cfg,cfg)

        return cfg

    @staticmethod
    def fromstring(cfg_str, file_format):
        """Generate config from config str.

        Args:
            cfg_str (str): Config str.
            file_format (str): Config file format corresponding to the
               config str. Only py/yml/yaml/json type are supported now!

        Returns:
            :obj:`Config`: Config obj.
        """
        if file_format not in ['.py', '.json', '.yaml', '.yml']:
            raise IOError('Only py/yml/yaml/json type are supported now!')
        if file_format != '.py' and 'dict(' in cfg_str:
            # check if users specify a wrong suffix for python
            warnings.warn(
                'Please check "file_format", the file format may be .py')
        with tempfile.NamedTemporaryFile(
                'w', encoding='utf-8', suffix=file_format,
                delete=False) as temp_file:
            temp_file.write(cfg_str)
            # on windows, previous implementation cause error
            # see PR 1077 for details
        cfg = Config.fromfile(temp_file.name)
        os.remove(temp_file.name)
        return cfg

    @staticmethod
    def auto_argparser(description=None):
        """Generate argparser from config file automatically (experimental)"""
        partial_parser = ArgumentParser(description=description)
        partial_parser.add_argument('config', help='config file path')
        cfg_file = partial_parser.parse_known_args()[0].config
        cfg = Config.fromfile(cfg_file)
        parser = ArgumentParser(description=description)
        parser.add_argument('config', help='config file path')
        add_args(parser, cfg)
        return parser, cfg

    def __init__(self, cfg_dict=None, cfg_text=None, filename=None):
        if cfg_dict is None:
            cfg_dict = dict()
        elif not isinstance(cfg_dict, dict):
            raise TypeError('cfg_dict must be a dict, but '
                            f'got {type(cfg_dict)}')
        for key in cfg_dict:
            if key in RESERVED_KEYS:
                raise KeyError(f'{key} is reserved for config file')

        super(Config, self).__setattr__('_cfg_dict', ConfigDict(cfg_dict))
        super(Config, self).__setattr__('_filename', filename)
        if cfg_text:
            text = cfg_text
        elif filename:
            with open(filename, 'r') as f:
                text = f.read()
        else:
            text = ''
        super(Config, self).__setattr__('_text', text)

    @property
    def filename(self):
        return self._filename

    @property
    def text(self):
        return self._text

    @property
    def pretty_text(self):

        indent = 4

        def _indent(s_, num_spaces):
            s = s_.split('\n')
            if len(s) == 1:
                return s_
            first = s.pop(0)
            s = [(num_spaces * ' ') + line for line in s]
            s = '\n'.join(s)
            s = first + '\n' + s
            return s

        def _format_basic_types(k, v, use_mapping=False):
            if isinstance(v, str):
                v_str = f"'{v}'"
            else:
                v_str = str(v)

            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent)

            return attr_str

        def _format_list(k, v, use_mapping=False):
            # check if all items in the list are dict
            if all(isinstance(_, dict) for _ in v):
                v_str = '[\n'
                v_str += '\n'.join(
                    f'dict({_indent(_format_dict(v_), indent)}),'
                    for v_ in v).rstrip(',')
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: {v_str}'
                else:
                    attr_str = f'{str(k)}={v_str}'
                attr_str = _indent(attr_str, indent) + ']'
            else:
                attr_str = _format_basic_types(k, v, use_mapping)
            return attr_str

        def _contain_invalid_identifier(dict_str):
            contain_invalid_identifier = False
            for key_name in dict_str:
                contain_invalid_identifier |= \
                    (not str(key_name).isidentifier())
            return contain_invalid_identifier

        def _format_dict(input_dict, outest_level=False):
            r = ''
            s = []

            use_mapping = _contain_invalid_identifier(input_dict)
            if use_mapping:
                r += '{'
            for idx, (k, v) in enumerate(input_dict.items()):
                is_last = idx >= len(input_dict) - 1
                end = '' if outest_level or is_last else ','
                if isinstance(v, dict):
                    v_str = '\n' + _format_dict(v)
                    if use_mapping:
                        k_str = f"'{k}'" if isinstance(k, str) else str(k)
                        attr_str = f'{k_str}: dict({v_str}'
                    else:
                        attr_str = f'{str(k)}=dict({v_str}'
                    attr_str = _indent(attr_str, indent) + ')' + end
                elif isinstance(v, list):
                    attr_str = _format_list(k, v, use_mapping) + end
                else:
                    attr_str = _format_basic_types(k, v, use_mapping) + end

                s.append(attr_str)
            r += '\n'.join(s)
            if use_mapping:
                r += '}'
            return r

        cfg_dict = self._cfg_dict.to_dict()
        text = _format_dict(cfg_dict, outest_level=True)
        # copied from setup.cfg
        yapf_style = dict(
            based_on_style='pep8',
            blank_line_before_nested_class_or_def=True,
            split_before_expression_after_opening_paren=True)
        #text, _ = FormatCode(text, style_config=yapf_style, verify=True)
        text, _ = FormatCode(text, style_config=yapf_style)

        return text

    def __repr__(self):
        return f'Config (path: {self.filename}): {self._cfg_dict.__repr__()}'

    def __len__(self):
        return len(self._cfg_dict)

    def __getattr__(self, name):
        return getattr(self._cfg_dict, name)

    def __getitem__(self, name):
        return self._cfg_dict.__getitem__(name)

    def __setattr__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setattr__(name, value)

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)
        self._cfg_dict.__setitem__(name, value)

    def __iter__(self):
        return iter(self._cfg_dict)

    def __getstate__(self):
        return (self._cfg_dict, self._filename, self._text)

    def __copy__(self):
        cls = self.__class__
        other = cls.__new__(cls)
        other.__dict__.update(self.__dict__)

        return other

    def __deepcopy__(self, memo):
        cls = self.__class__
        other = cls.__new__(cls)
        memo[id(self)] = other

        for key, value in self.__dict__.items():
            super(Config, other).__setattr__(key, copy.deepcopy(value, memo))

        return other

    def __setstate__(self, state):
        _cfg_dict, _filename, _text = state
        super(Config, self).__setattr__('_cfg_dict', _cfg_dict)
        super(Config, self).__setattr__('_filename', _filename)
        super(Config, self).__setattr__('_text', _text)

    def dump(self, file=None):
        cfg_dict = super(Config, self).__getattribute__('_cfg_dict').to_dict()
        if self.filename is None or self.filename.endswith('.py'):
            if file is None:
                return self.pretty_text
            else:
                with open(file, 'w', encoding='utf-8') as f:
                    f.write(self.pretty_text)
        else:
            import mmcv
            if file is None:
                file_format = self.filename.split('.')[-1]
                return mmcv.dump(cfg_dict, file_format=file_format)
            else:
                mmcv.dump(cfg_dict, file)

    def merge_from_dict(self, options, allow_list_keys=True):
        """Merge list into cfg_dict.

        Merge the dict parsed by MultipleKVAction into this cfg.

        Examples:
            >>> options = {'model.backbone.depth': 50,
            ...            'model.backbone.with_cp':True}
            >>> cfg = Config(dict(model=dict(backbone=dict(type='ResNet'))))
            >>> cfg.merge_from_dict(options)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(
            ...     model=dict(backbone=dict(depth=50, with_cp=True)))

            >>> # Merge list element
            >>> cfg = Config(dict(pipeline=[
            ...     dict(type='LoadImage'), dict(type='LoadAnnotations')]))
            >>> options = dict(pipeline={'0': dict(type='SelfLoadImage')})
            >>> cfg.merge_from_dict(options, allow_list_keys=True)
            >>> cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
            >>> assert cfg_dict == dict(pipeline=[
            ...     dict(type='SelfLoadImage'), dict(type='LoadAnnotations')])

        Args:
            options (dict): dict of configs to merge from.
            allow_list_keys (bool): If True, int string keys (e.g. '0', '1')
              are allowed in ``options`` and will replace the element of the
              corresponding index in the config if the config is a list.
              Default: True.
        """
        option_cfg_dict = {}
        for full_key, v in options.items():
            d = option_cfg_dict
            key_list = full_key.split('.')
            for subkey in key_list[:-1]:
                d.setdefault(subkey, ConfigDict())
                d = d[subkey]
            subkey = key_list[-1]
            d[subkey] = v

        cfg_dict = super(Config, self).__getattribute__('_cfg_dict')
        super(Config, self).__setattr__(
            '_cfg_dict',
            Config._merge_a_into_b(
                option_cfg_dict, cfg_dict, allow_list_keys=allow_list_keys))


    @staticmethod
    def set_key_vals(root_cfg,cfg,parent_keys=""):
        '''
        define variable by: NAME = value
        use variable by: $$NAME
    
        Example:
        define:
        CLASSES_NUM = classes_num
        use:
        classes_num = $$CLASSES_NUM
        '''
        TAG = "$$"
        W_TAG = "$"
    
        variable_dict = root_cfg.get("variable",{})
        def _get_variable(key):
            if key in root_cfg:
                return root_cfg[key]
            elif key in variable_dict:
                return variable_dict[key]
            return None
    
        def get_variable(key):
            if "(" in key and ")" in key:
                key = TAG+key
                try:
                    pattern = re.compile("\$\$\(.*?\)")
                    res = pattern.finditer(key)
                    for rg in list(res)[::-1]:
                        tmp_key = key[rg.start():rg.end()]
                        tmp_key = tmp_key[3:-1]
                        key = key[:rg.start()]+str(_get_variable(tmp_key))+key[rg.end():]
                    return eval(key)
                except:
                    print(f"ERROR: get value for key {key} faild.")
                    return None
            if key in root_cfg:
                return root_cfg[key]
            elif key in variable_dict:
                return variable_dict[key]
            return None
    
        for k,v in cfg.items():
            if isinstance(v,dict):
                all_keys = parent_keys+"."+str(k)
                Config.set_key_vals(root_cfg,v,all_keys)
                continue
            elif isinstance(v,(list,tuple)):
                for i,cv in enumerate(v):
                    if not isinstance(cv,(dict)):
                        continue
                    all_keys = parent_keys+f".[{i}]"
                    Config.set_key_vals(root_cfg,cv,all_keys)
                    continue
    
            if not isinstance(v,(str,bytes)):
                continue
    
            if v.startswith(TAG):
                r_key = v[len(TAG):]
                n_v = get_variable(r_key)
                if n_v is None:
                    msg = f"Find value {v} in cfg faild."
                    print(msg)
                    raise RuntimeError(msg)
                cfg[k] = n_v
                all_keys = parent_keys+"."+str(k)
                print(f"Update cfg{all_keys} from {v} to {n_v}")
            elif v.startswith(W_TAG):
                all_keys = parent_keys+"."+str(k)
                print(f"WRANING: ambiguous value {v} for cfg{all_keys}")

    @staticmethod
    def set_sibling_key_vals(cfg,parent_keys=""):
        '''
        用于引用兄弟的值
        define variable by: NAME = value
        use variable by: .$$NAME
    
        Example:
        define:
            {
                a=1234,
                {
                    b=321,   #定义
                    c=".$$b", #使用
                }
            }
        
        $$(NAME)+100
        $$(NAME1)+$$(NAME2)
        $$(NAME1)*2
        '''
        TAG = "$$."
        W_TAG = "$."
    
        def _get_key_value(data,key):
            '''
            data: root config
            '''
            if "(" in key and ")" in key:
                #如果()包含了值，使用eval评估
                key = TAG+key
                try:
                    pattern = re.compile("\$\$\(\..*?\)")
                    res = pattern.finditer(key)
                    for rg in list(res)[::-1]:
                        tmp_key = key[rg.start():rg.end()]
                        tmp_key = tmp_key[4:-1]
                        key = key[:rg.start()]+str(_get_key_value(data,tmp_key))+key[rg.end():]
                    return eval(key)
                except:
                    print(f"ERROR: get value for key {key} faild.")
                    return None
            v = data.get(key,None)  #直接从root config中获取相应的值
            if isinstance(v,(str,bytes)):  #处理嵌套引用问题
                if v.startswith(TAG):
                    r_key = v[len(TAG):]
                    return _get_key_value(data,r_key)
            return v
    
        for k,v in cfg.items():
            if isinstance(v,dict):
                all_keys = parent_keys+"."+str(k)
                Config.set_sibling_key_vals(v,all_keys)
                continue
            elif isinstance(v,(list,tuple)):
                for i,cv in enumerate(v):
                    if not isinstance(cv,(dict)):
                        continue
                    all_keys = parent_keys+f".[{i}]"
                    Config.set_sibling_key_vals(cv,all_keys)
                    continue
    
            if not isinstance(v,(str,bytes)):
                continue
    
            if v.startswith(TAG):
                r_key = v[len(TAG):]
                n_v = _get_key_value(cfg,r_key)
                if n_v is None:
                    msg = f"Find value {v} in cfg faild."
                    print(msg)
                    raise RuntimeError(msg)
                cfg[k] = n_v
                all_keys = parent_keys+"."+str(k)
                print(f"Update cfg{all_keys} from {v} to {n_v}")
            elif v.startswith(W_TAG):
                all_keys = parent_keys+"."+str(k)
                print(f"WRANING: ambiguous value {v} for cfg{all_keys}")
    
    @staticmethod
    def get_base_file_list(file_path):
        file_path = osp.abspath(file_path)
        cur_base_file_list,*_ = Config.simple_get_base_file_list(file_path)
        base_file_list = []
        for f in cur_base_file_list[::-1]:
            cur_list = Config.get_base_file_list(f)
            base_file_list = cur_list+[f]+base_file_list
        
        res_base_file_list = []
        for x in base_file_list:
            if x not in res_base_file_list:
                res_base_file_list = res_base_file_list+[x]
        
        return res_base_file_list


    @staticmethod
    def simple_get_base_file_list(file_path):
        if not isinstance(file_path,(list,tuple)):
            with open(file_path,"r") as f:
                lines = f.readlines()
        else:
            lines = file_path
            file_path = None
        
        start_no = -1
        end_no = -1
        find_bracket = False
        find_bias = False
        for i,l in enumerate(lines):
            l = l.rstrip()
            if BASE_KEY in l and start_no<0:
                start_no = i
                if "[" in l:
                    find_bracket = True
                elif "\\" in l:
                    find_bias = True
            if start_no>=0:
                if find_bracket:
                    if "]" in l:
                        end_no = i
                        break
                elif find_bias:
                    if "\\" not in l:
                        end_no = i
                        break
                else:
                    end_no = i
                    break
        
        if start_no>=0 and end_no>=0:
            data = "\n".join(lines[start_no:end_no+1])
            data = Config._simple_str2dict(data)
            if BASE_KEY in data:
                data = data[BASE_KEY]
                if not isinstance(data,(list,tuple)):
                    data = [data]
                if file_path is not None:
                    dir_name = osp.dirname(file_path)
                    data = [osp.abspath(osp.join(dir_name,x)) for x in data]
            else:
                data = []
            return data,start_no,end_no
        else:
            return [],None,None

    
    @staticmethod
    def _insert_ph_values2str(data,ph_values):
        data = data.split("\n")
        new_data = []
        import_lines = -1
        _,start_no,end_no = Config.simple_get_base_file_list(data)
        if end_no is not None:
            import_lines = end_no
        for i,d in enumerate(data):
            if "import " in d and i>import_lines:
                import_lines = i
        ph_lines = [k+" = "+Config.to_placeholder_str(v) for k,v in ph_values.items()]
        new_data = data[:import_lines+1]+ph_lines+data[import_lines+1:]
        new_data = "\n".join(new_data)
        return new_data
        
    @staticmethod
    def _remove_ph_wrap(data,ph_values={}):
        if ph_values is None:
            ph_values = {}
        ph_regexp = Config.ph_regexp
        base_vars = set(re.findall(ph_regexp, data))
        for base_var in base_vars:
            if base_var in ph_values:
                value = Config.to_placeholder_str(ph_values[base_var])
            else:
                value = base_var
            regexp = Config.ph_regexp_pre+base_var
            data = re.sub(regexp, value, data)
        return data

    @staticmethod
    def _simple_str2dict(data,tmp_dir=None,ph_values={}):

        if ph_values is not None and len(ph_values)>0:
            data = Config._insert_ph_values2str(data,ph_values)
            data = Config._remove_ph_wrap(data)

        temp_config_file = tempfile.NamedTemporaryFile(
                dir=tmp_dir, suffix=".py")

        temp_config_name = osp.basename(temp_config_file.name)
        temp_config_dir = osp.dirname(temp_config_file.name)

        sys.path.insert(0, temp_config_dir)
        temp_module_name = osp.splitext(temp_config_name)[0]
        with open(temp_config_file.name,"w") as f:
            f.write(data)
        invalidate_caches()
        mod = import_module(temp_module_name)
        #sys.path.pop(0)
        cfg_dict = {
                    name: value
                    for name, value in mod.__dict__.items()
                    if not name.startswith('__')
                    and not isinstance(value, types.ModuleType)
                    and not isinstance(value, types.FunctionType)
                }
                # delete imported module
        del sys.modules[temp_module_name]
        temp_config_file.close()

        return cfg_dict
                

    @staticmethod
    def get_placeholder_names(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            config_file = f.read()
        ph_regexp = Config.ph_regexp
        base_vars = set(re.findall(ph_regexp, config_file))
        res = []
        for base_var in base_vars:
            res.append(base_var)
        return res

    @staticmethod
    def get_placeholder_values(filename):
        '''
        使用vars.NAME的方式引用其它变量
        '''
        filename = osp.abspath(osp.expanduser(filename))
        basename = osp.basename(osp.splitext(filename)[0])
        dirname = osp.dirname(filename)
        dirbasename = osp.basename(dirname)
        pre_ph_values = dict(filename=filename,basename=basename,dirname=dirname,dirbasename=dirbasename)

        ph_names = Config.get_placeholder_names(filename)
        base_files = Config.get_base_file_list(filename)
        print(f"Base file list:")
        for bf in base_files:
            print(f"base file: {bf}")
            ph_names += Config.get_placeholder_names(bf)
        ph_names = list(set(ph_names))

        ph_values = dict(zip(ph_names,[None]*len(ph_names)))
        ph_values.update(pre_ph_values)
        all_files = base_files+[filename]
        for bf in all_files:
            with open(bf,"r") as f:
                config_file = f.read()
            cur_data_dict = Config._simple_str2dict(config_file,ph_values=ph_values)
            for k,v in cur_data_dict.items():
                if k in ph_values and v is not None:
                    ph_values[k] = v
        return ph_values
    
    @staticmethod
    def to_placeholder_str(v):
        if isinstance(v,str):
            return "\""+v+"\""
        
        return str(v)

    @staticmethod
    def remove_tmp_value(cfg_dict):
        if isinstance(cfg_dict,dict):
            for k in list(cfg_dict.keys()):
                if not isinstance(k,str):
                    continue
                if k.endswith("_"):
                    cfg_dict.pop(k)
            for k,v in cfg_dict.items():
                Config.remove_tmp_value(v)
        elif isinstance(cfg_dict,(list,tuple)):
            cfg_dict = [Config.remove_tmp_value(x) for x in cfg_dict]

        return cfg_dict

    @staticmethod
    @wmlu.no_throw
    def get_value_in_cfgs(cfgs,attr,default_value=None):
        for cfg in cfgs:
            if cfg is not None and hasattr(cfg,attr):
                return cfg.get(attr,default_value)
        return default_value

class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val):
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple: The expanded list or tuple from the string.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)
