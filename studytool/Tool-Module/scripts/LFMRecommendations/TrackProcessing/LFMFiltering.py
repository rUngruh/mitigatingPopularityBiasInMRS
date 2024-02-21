# -*- coding: utf-8 -*-


def get_valid_spids(spids, spid_ids):
    # get all valid spids (those that are included in the LFM dataset)
    all_valids = set(spid_ids['uri'])
    valid_spids = list(set(spids) & all_valids)
    return valid_spids