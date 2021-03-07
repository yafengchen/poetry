#!/usr/bin/env python

from poembase_3 import PoemBase

class Poem(PoemBase):

    def __init__(self, form='short', config='config/sylvia.json'):
        super().__init__(form=form, config=config)

