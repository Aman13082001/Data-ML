"""Tokenisation shared by the classifiers: normalise, expand abbreviations,
light stemming, concept tags from a generic lexicon (resources/)."""
import os
import re

import yaml

from .clean import norm_text
from .config import REPO

_STOP = set("""A AN AND THE OF FOR TO IN ON WITH BY OR AT FROM AS PER X NO NOT
INCL INCLUDING OTHER OTHERS MISC GENERAL VARIOUS ETC EA EACH PCS PC UNIT UNITS
DE LA EL LOS LAS DEL Y PARA CON EN LE LES DES ET DU DER DIE DAS UND FUR MIT
SUPPLY SUPPLIES ITEM ITEMS""".split())


def _load(name):
    p = os.path.join(REPO, "resources", name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class Lexicon:
    def __init__(self):
        ab = _load("abbreviations.yaml")
        self.abbr = {k.upper(): str(v).upper() for k, v in (ab or {}).items()}
        lx = _load("lexicon.yaml") or {}
        self.word2groups = {}      # single word -> groups
        self.phrases = []          # (frozenset of words, group): all words must be present
        for g, words in lx.items():
            g = g.split("__")[0]           # 'fastener__intl' adds words to 'fastener'
            for w in words:
                toks = self.tokens(str(w), expand=False)
                if len(toks) == 1:
                    self.word2groups.setdefault(toks[0], set()).add(g)
                elif toks:
                    self.phrases.append((frozenset(toks), g))

    @staticmethod
    def stem(w):
        if len(w) <= 4 or w.isdigit():
            return w
        # plurals only: 'BOLTER' (a machine) must stay apart from 'BOLT' (an item)
        if w.endswith("IES") and len(w) > 5:
            return w[:-3] + "Y"
        if w.endswith(("SSES", "XES", "CHES", "SHES")):
            return w[:-2]
        if w.endswith("S") and not w.endswith(("SS", "US", "IS")):
            return w[:-1]
        return w

    def tokens(self, text, expand=True):
        t = norm_text(text)
        words = re.findall(r"[A-Z][A-Z]+", t)
        out = []
        for w in words:
            if expand and w in self.abbr:
                for x in self.abbr[w].split():
                    if x not in _STOP:
                        out.append(self.stem(x))
                continue
            if w in _STOP:
                continue
            out.append(self.stem(w))
        return out

    def concepts(self, toks):
        gs = set()
        ts = set(toks)
        for t in ts:
            gs |= self.word2groups.get(t, set())
        for ph, g in self.phrases:
            if ph <= ts:
                gs.add(g)
        return sorted(gs)


_LEX = None


def lexicon():
    global _LEX
    if _LEX is None:
        _LEX = Lexicon()
    return _LEX
