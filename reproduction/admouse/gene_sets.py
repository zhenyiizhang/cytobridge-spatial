"""Gene sets used for the AD module-score comparisons."""
GENE_SETS = {
    "DAM_microglia": [
        "Trem2", "Apoe", "Tyrobp", "C1qa", "C1qb", "C1qc", "Itgax",
        "Spp1", "Lpl", "Cst7", "Cd74", "Gpnmb", "Lgals3", "Csf1r",
        "Hexb", "Ctsb", "Cd68", "Aif1",
    ],
    "DAM_Lipid_Metabolism": [
        "Trem2", "Apoe", "Lpl", "Gpnmb", "Lgals3", "Cst7", "Fabp5", "Abca1",
    ],
    "Lysosome_Phagosome": [
        "Cd68", "Ctsb", "Ctss", "Ctsd", "Lamp1", "Hexb", "Tyrobp", "Trem2",
    ],
    "AB_Clearance_Endolysosomal": [
        "Picalm", "Lrp1", "Bin1", "Cltc", "Clta", "Apoe", "Trem2", "Axl",
        "Mertk", "Ctsb", "Cd68", "Sort1",
    ],
    "Inflammation_Complement": [
        "C1qa", "C1qb", "C1qc", "C3", "C4b", "Itgam", "Itgb2", "Itgax",
        "Tyrobp", "C3ar1", "C5ar1", "Ctsb", "Csf1r",
    ],
    "Astrocyte_Reactive": [
        "Gfap", "Aqp4", "Vim", "Serpina3n", "C3", "Cxcl10", "S100b",
        "Cd44", "Stat3", "Lcn2", "Socs3", "C4b", "H2-D1",
    ],
    "SPP1_CD44_axis": ["Spp1", "Cd44", "Cxcl10", "Csf1", "Csf1r"],
    "Myelination_Oligo": [
        "Olig2", "Sox10", "Myrf", "Plp1", "Mobp", "Mbp", "Mog", "Opalin",
        "Cnp", "Mag", "Mal", "Ermn", "Nfasc", "Tspan2",
    ],
    "Endothelial_BBB": ["Pecam1", "Cldn5", "Kdr", "Flt1", "Mfsd2a", "Emcn", "Klf2"],
    "Antigen_Presentation_MHCII": [
        "Cd74", "H2-Ab1", "H2-Aa", "H2-D1", "Ifi30", "Lgmn", "Csf1r",
    ],
}
