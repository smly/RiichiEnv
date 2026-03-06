export const YAKU_MAP: { [key: number]: string } = {
    1: "Menzen Tsumo", 2: "Riichi", 3: "Chankan", 4: "Rinshan Kaihou", 5: "Haitei Raoyue", 6: "Houtei Raoyui",
    7: "Haku", 8: "Hatsu", 9: "Chun", 10: "Jikaze (Seat Wind)", 11: "Bakaze (Round Wind)",
    12: "Tanyao", 13: "Iipeiko", 14: "Pinfu", 15: "Chanta", 16: "Ittsu", 17: "Sanshoku Doujun",
    18: "Double Riichi", 19: "Sanshoku Doukou", 20: "Sankantsu", 21: "Toitoi", 22: "San Ankou",
    23: "Shousangen", 24: "Honroutou", 25: "Chiitoitsu", 26: "Junchan", 27: "Honitsu",
    28: "Ryanpeiko", 29: "Chinitsu", 30: "Ippatsu", 31: "Dora", 32: "Akadora", 33: "Ura Dora",
    35: "Tenhou", 36: "Chiihou", 37: "Daisangen", 38: "Suuankou", 39: "Tsuu Iisou",
    40: "Ryuu Iisou", 41: "Chinroutou", 42: "Kokushi Musou", 43: "Shousuushii", 44: "Suukantsu",
    45: "Chuuren Poutou", 47: "Junsei Chuuren Poutou", 48: "Suuankou Tanki", 49: "Kokushi Musou 13-wait",
    34: "Nukidora",
    50: "Daisuushii"
};

export const COLORS = {
    boardBackground: '#39416e',
    modalBackground: '#0f1744',
    tableHeaderBackground: '#0f1744',
    centerInfoBackground: '#13151b',

    highlightBoard: '#0f1744',
    highlightButton: '#2a3c85',
    activeButtonBg: '#0f1744',

    tableBorder: '#2a3c85',
    text: '#ffffff',
    activePlayerBar: '#ffd700',

    riverContainer: '#fff',

    callChiiBg: 'rgba(34, 139, 34, 0.85)',
    callPonBg: 'rgba(30, 80, 180, 0.85)',
    callKanBg: 'rgba(120, 50, 180, 0.85)',
    callReachBg: 'rgba(220, 120, 20, 0.90)',
    callHoraBg: 'rgba(200, 30, 30, 0.90)',
    callDefaultBg: 'rgba(0, 0, 0, 0.6)',
};

export const CALL_TYPES: { [key: string]: { label: string; cssClass?: string } } = {
    chi:       { label: 'Chii',  cssClass: 'call-chii' },
    pon:       { label: 'Pon',   cssClass: 'call-pon' },
    kan:       { label: 'Kan',   cssClass: 'call-kan' },
    ankan:     { label: 'Kan',   cssClass: 'call-kan' },
    daiminkan: { label: 'Kan',   cssClass: 'call-kan' },
    kakan:     { label: 'Kan',   cssClass: 'call-kan' },
    reach:     { label: 'Riichi', cssClass: 'call-reach' },
    kita:      { label: 'Pei',    cssClass: 'call-kan' },
};
