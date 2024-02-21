# -*- coding: utf-8 -*-


def uris_to_spids(uris):
    # transform the whole uri into the spotify id
    return [uri.split(':')[-1] for uri in uris]

def spids_to_ids(spids, spids_ids):
    # transform the spotify ids into the track ids
    matching_rows = spids_ids[spids_ids['uri'].isin(spids)]
    track_ids = matching_rows.groupby('uri')['track_id'].first().tolist()
    return track_ids



def ids_to_spids(ids, spids_ids):
    # transform the ids into the spotify ids
    return [spids_ids[spids_ids['track_id']==tid]['uri'].item() for tid in ids]

