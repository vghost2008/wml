from abc import ABCMeta, abstractmethod


class WBaseMaskLike(metaclass=ABCMeta):
    HORIZONTAL = 'horizontal'
    VERTICAL =  'vertical'
    DIAGONAL =  'diagonal'
    def __init__(self) -> None:
        pass