import scipy.sparse as sp
import dgl
import numpy as np
from pathlib import Path


META_PATH_NAMES = {
    "amazon": {
        "user": ("UIU", "UIBIU", "UICIU"),
        "item": ("IBI", "ICI", "IVI"),
    },
    "yelp": {
        "user": ("UU", "UBU", "UCU"),
        "item": ("BCiB", "BCaB", "BUB"),
    },
    "lastfm": {
        "user": ("UU", "UATAU", "UAU"),
        "item": ("AA", "ATA", "AUA"),
    },
    "movielens-1m": {
        "user": ("UMU", "UOU", "UAU"),
        "item": ("MUM", "MGM", "MM"),
    },
    "douban-book": {
        "user": ("UU", "UGU", "ULU"),
        "item": ("BAB", "BYB", "BPB"),
    },
    "douban-movie": {
        "user": ("UUUUU", "UGU", "UMU"),
        "item": ("MAM", "MDM", "MTM"),
    },
}


def get_meta_path_names(dataset):
    try:
        names = META_PATH_NAMES[dataset]
    except KeyError as error:
        raise ValueError(f"Unsupported dataset: {dataset}") from error
    return names["user"], names["item"]


def load_data(dataset, device, data_root="data"):
    dataset_root = Path(data_root) / dataset

    def load_npz(name):
        path = dataset_root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing meta-path graph: {path}")
        return sp.load_npz(path)

    if dataset == 'douban-movie':
        uuuuu_graph = load_npz('uuuuu.npz')
        ugu_graph = load_npz('ugu_processed.npz')
        umu_graph = load_npz('umu_processed.npz')

        mam_graph = load_npz('mam.npz')
        mdm_graph = load_npz('mdm.npz')
        mtm_graph = load_npz('mtm_processed.npz')

        uuuuu_graph = get_graph(uuuuu_graph).to(device)
        ugu_graph = get_graph(ugu_graph).to(device)
        umu_graph = get_graph(umu_graph).to(device)

        mam_graph = get_graph(mam_graph).to(device)
        mdm_graph = get_graph(mdm_graph).to(device)
        mtm_graph = get_graph(mtm_graph).to(device)

        user_mps = [uuuuu_graph,ugu_graph,umu_graph,]
        item_mps = [mam_graph,mdm_graph,mtm_graph,]

    elif dataset == 'lastfm':
        uu_graph = load_npz('uu.npz')
        uatau_graph = load_npz('uatau_processed.npz')
        uau_graph = load_npz('uau_processed.npz')

        aa_graph = load_npz('aa.npz')
        ata_graph = load_npz('ata_processed.npz')
        aua_graph = load_npz('aua_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        uatau_graph = get_graph(uatau_graph).to(device)
        uau_graph = get_graph(uau_graph).to(device)

        aa_graph = get_graph(aa_graph).to(device)
        ata_graph = get_graph(ata_graph).to(device)
        aua_graph = get_graph(aua_graph).to(device)

        user_mps = [
            uu_graph,
            uatau_graph,
            uau_graph,
        ]
        item_mps = [
            aa_graph,
            ata_graph,
            aua_graph,
        ]

    elif dataset == 'yelp':
        uu_graph = load_npz('uu.npz')
        ubu_graph = load_npz('ubu_processed.npz')
        ucu_graph = load_npz('ucu_processed.npz')

        bcib_graph = load_npz('bcib_processed.npz')
        bcab_graph = load_npz('bcab_processed.npz')
        bub_graph = load_npz('bub_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        ubu_graph = get_graph(ubu_graph).to(device)
        ucu_graph = get_graph(ucu_graph).to(device)

        bcib_graph = get_graph(bcib_graph).to(device)
        bcab_graph = get_graph(bcab_graph).to(device)
        bub_graph = get_graph(bub_graph).to(device)

        user_mps = [
            uu_graph,
            ubu_graph,
            ucu_graph,
        ]
        item_mps = [
            bcib_graph,
            bcab_graph,
            bub_graph,
        ]

    elif dataset == 'douban-book':
        uu_graph = load_npz('uu.npz')
        ugu_graph = load_npz('ugu_processed.npz')
        ulu_graph = load_npz('ulu_processed.npz')

        bab_graph = load_npz('bab.npz')
        byb_graph = load_npz('byb_processed.npz')
        bpb_graph = load_npz('bpb_processed.npz')

        uu_graph = get_graph(uu_graph).to(device)
        ugu_graph = get_graph(ugu_graph).to(device)
        ulu_graph = get_graph(ulu_graph).to(device)

        bab_graph = get_graph(bab_graph).to(device)
        byb_graph = get_graph(byb_graph).to(device)
        bpb_graph = get_graph(bpb_graph).to(device)

        user_mps = [
            uu_graph,
            ugu_graph,
            ulu_graph,
        ]
        item_mps = [
            bab_graph,
            byb_graph,
            bpb_graph,
        ]

    elif dataset == 'amazon':
        # uibiu_graph = sp.load_npz('data/amazon/uibiu_processed.npz')
        # # uiviu_graph = sp.load_npz('data/amazon/uiviu.npz')
        # uiu_graph = sp.load_npz('data/amazon/uiu_processed.npz')
        # ibi_graph = sp.load_npz('data/amazon/ibi_processed.npz')
        # # ivi_graph = sp.load_npz('data/amazon/ivi.npz')
        # # iui_graph = sp.load_npz('data/amazon/iui.npz')
        # ici_graph = sp.load_npz('data/amazon/ici_processed.npz')

        uiu_graph = load_npz('uiu_processed.npz')
        uibiu_graph = load_npz('uibiu_processed.npz')
        uiciu_graph = load_npz('uiciu_processed.npz').tocoo()

        ibi_graph = load_npz('ibi_processed.npz')
        ici_graph = load_npz('ici_processed.npz')
        ivi_graph = load_npz('ivi.npz')


        # uibiu_graph = get_graph(uibiu_graph).to(device)
        # # uiviu_graph = get_graph(uiviu_graph).to(device)
        # uiu_graph = get_graph(uiu_graph).to(device)
        # ibi_graph = get_graph(ibi_graph).to(device)
        # # ivi_graph = get_graph(ivi_graph).to(device)
        # # iui_graph = get_graph(iui_graph).to(device)
        # ici_graph = get_graph(ici_graph).to(device)

        uiu_graph = get_graph(uiu_graph).to(device)
        uibiu_graph = get_graph(uibiu_graph).to(device)
        uiciu_graph = get_graph(uiciu_graph).to(device)

        ibi_graph = get_graph(ibi_graph).to(device)
        ici_graph = get_graph(ici_graph).to(device)
        ivi_graph = get_graph(ivi_graph).to(device)

        #user_mps, item_mps = [uibiu_graph, uiu_graph], [ibi_graph, ici_graph]  #

        user_mps = [uiu_graph, uibiu_graph, uiciu_graph]
        item_mps = [ibi_graph, ici_graph, ivi_graph]



    elif dataset == 'movielens-1m':
        umu_graph = load_npz('umu_processed.npz')
        uou_graph = load_npz('uou_processed.npz')
        uau_graph = load_npz('uau_processed.npz')

        mum_graph = load_npz('mum_processed.npz')
        mgm_graph = load_npz('mgm_processed.npz')
        mm_graph = load_npz('mm_processed.npz')

        umu_graph = get_graph(umu_graph).to(device)
        uou_graph = get_graph(uou_graph).to(device)
        uau_graph = get_graph(uau_graph).to(device)

        mum_graph = get_graph(mum_graph).to(device)
        mgm_graph = get_graph(mgm_graph).to(device)
        mm_graph = get_graph(mm_graph).to(device)

        user_mps = [
            umu_graph,
            uou_graph,
            uau_graph,
        ]
        item_mps = [
            mum_graph,
            mgm_graph,
            mm_graph,
        ]

    user_names, item_names = get_meta_path_names(dataset)
    if len(user_mps) != len(user_names) or len(item_mps) != len(item_names):
        raise RuntimeError(
            "Loaded graph counts do not match the manuscript meta-path names."
        )
    return user_mps, item_mps

def get_graph(graph):
    graph = graph.tocoo()
    new_row = graph.row
    new_col = graph.col
    g = dgl.graph((new_row, new_col), num_nodes=graph.shape[0])
    return g
