#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional, Any
import copy
import inspect


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.

    To create a registry (e.g. a backbone registry):

    .. code-block:: python

        BACKBONE_REGISTRY = Registry('BACKBONE')

    To register an object:

    .. code-block:: python

        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...

    Or:

    .. code-block:: python

        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    def _do_register(self, name: str, obj: object) -> None:
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name
        )
        self._obj_map[name] = obj

    def register(self, obj: object = None,name=None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object,name=name) -> object:
                if name is None:
                    name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret

    def __contains__(self,v):
        return v in self._obj_map
    
    def build(self,cfg):
        if isinstance(cfg,(list,tuple)):
            return [self.build(ccfg) for ccfg in cfg]
            
        if 'type' not in cfg:
            if 'default' in cfg and 'cfgs' in cfg:
                default = cfg['default']
                cfgs = []
                for lcfg1 in cfg['cfgs']:
                    lcfg = copy.deepcopy(default)
                    lcfg.merge_from_dict(lcfg1)
                    cfgs.append(lcfg)
                return self.__build__(cfgs)
        return self.__build__(cfg)

    def __build__(self,cfg):
        return self.build_from_cfg(cfg,self)


    @staticmethod
    def expand2list(cfg,num):
        if cfg is None:
            return None
        if 'default' in cfg and 'cfgs' in cfg:
            default = cfg['default']
            cfgs = []
            for lcfg1 in cfg['cfgs']:
                lcfg = copy.deepcopy(default)
                lcfg.merge_from_dict(lcfg1)
                cfgs.append(lcfg)
            return cfgs

        if not isinstance(cfg, list):
            cfgs = [
                cfg for _ in range(num)
            ]
            return cfgs
        
        return cfg


    @staticmethod
    def build_from_cfg(cfg: Dict,
                       registry: 'Registry',
                       default_args: Optional[Dict] = None) -> Any:

        if not isinstance(cfg, dict):
            raise TypeError(f'cfg must be a dict, but got {type(cfg)}')
        if 'type' not in cfg:
            if default_args is None or 'type' not in default_args:
                raise KeyError(
                    '`cfg` or `default_args` must contain the key "type", '
                    f'but got {cfg}\n{default_args}')
        if not isinstance(registry, Registry):
            raise TypeError('registry must be an mmcv.Registry object, '
                            f'but got {type(registry)}')
        if not (isinstance(default_args, dict) or default_args is None):
            raise TypeError('default_args must be a dict or None, '
                            f'but got {type(default_args)}')
    
        args = cfg.copy()
    
        if default_args is not None:
            for name, value in default_args.items():
                args.setdefault(name, value)
    
        obj_type = args.pop('type')
        if isinstance(obj_type, str):
            obj_cls = registry.get(obj_type)
            if obj_cls is None:
                raise KeyError(
                    f'{obj_type} is not in the {registry.name} registry')
        elif inspect.isclass(obj_type) or inspect.isfunction(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')
        try:
            return obj_cls(**args)
        except Exception as e:
            # Normal TypeError does not print class name.
            raise type(e)(f'{obj_cls.__name__}: {e}, cfg={cfg}')
