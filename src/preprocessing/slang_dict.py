# ============================================================
# src/preprocessing/slang_dict.py
# Kamus normalisasi slang & singkatan bahasa Indonesia informal
# 200+ entri — digunakan oleh normalizer.py
# ============================================================

# Format: "kata_slang": "kata_baku"
# Nilai string kosong ("") artinya kata dihapus (noise)

SLANG_DICT: dict[str, str] = {
    # ── Negasi (JANGAN HAPUS dari kamus — hanya normalisasi penulisan) ──
    "gk": "tidak", "gak": "tidak", "ga": "tidak", "g": "tidak",
    "ngga": "tidak", "nggak": "tidak", "nggk": "tidak",
    "tak": "tidak", "ga ada": "tidak ada",

    # ── Kata ganti ────────────────────────────────────────────
    "sy": "saya", "aq": "saya", "aku": "saya",
    "gw": "saya", "gue": "saya", "w": "saya",
    "km": "kamu", "lo": "kamu", "lu": "kamu",
    "dy": "dia", "doi": "dia",
    "mrk": "mereka", "kalian": "kalian",
    "kita": "kita",

    # ── Kata penghubung & depan ───────────────────────────────
    "yg": "yang", "yng": "yang",
    "dgn": "dengan", "dg": "dengan", "sama": "dengan",
    "krn": "karena", "karna": "karena", "krna": "karena",
    "utk": "untuk", "buat": "untuk", "tuk": "untuk",
    "tp": "tapi", "tpi": "tapi",
    "dr": "dari", "dr.": "dari",
    "ke": "ke", "pd": "pada", "spy": "supaya",
    "sdgkn": "sedangkan", "sdgkan": "sedangkan",
    "krg": "kurang", "ttg": "tentang",

    # ── Kata kerja umum ───────────────────────────────────────
    "bs": "bisa", "bsa": "bisa",
    "sdh": "sudah", "udh": "sudah", "udah": "sudah", "dah": "sudah",
    "blm": "belum", "blum": "belum",
    "lg": "lagi", "lgi": "lagi",
    "msh": "masih", "masi": "masih",
    "mau": "mau", "mo": "mau",
    "liat": "lihat", "lht": "lihat",
    "dpt": "dapat", "dpat": "dapat",
    "beli": "beli", "bli": "beli",
    "pake": "pakai", "pk": "pakai", "pke": "pakai",
    "kasih": "kasih", "ksh": "kasih",
    "dtg": "datang", "nyampe": "sampai", "smpe": "sampai",
    "kirim": "kirim", "krm": "kirim",
    "trima": "terima", "trma": "terima",
    "nanya": "tanya", "tny": "tanya",
    "bilang": "bilang", "blg": "bilang",

    # ── Kata benda umum ───────────────────────────────────────
    "brg": "barang", "brng": "barang",
    "pkt": "paket", "paket": "paket",
    "hrg": "harga",
    "klt": "kualitas", "qlts": "kualitas",
    "wkt": "waktu", "wktu": "waktu",
    "tgn": "tangan",
    "tmn": "teman",
    "org": "orang", "org2": "orang-orang",
    "bln": "bulan", "thn": "tahun", "hr": "hari",
    "mgg": "minggu",
    "jm": "jam",
    "no": "nomor", "nomer": "nomor",
    "tlp": "telepon", "hp": "handphone",

    # ── Kata sifat & keterangan ───────────────────────────────
    "bgt": "banget", "bngt": "banget", "bget": "banget",
    "bgd": "banget",
    "sgt": "sangat", "sngt": "sangat",
    "bnr": "benar", "bner": "benar", "bner2": "benar-benar",
    "ckp": "cukup",
    "lumayan": "lumayan",
    "mntul": "mantap betul", "mantul": "mantap betul",
    "mantap": "mantap", "mntap": "mantap",
    "keren": "keren", "krn2": "keren",
    "bagus": "bagus", "bgs": "bagus",
    "jelek": "jelek", "jlk": "jelek",
    "murah": "murah", "mrh": "murah",
    "mahal": "mahal", "mhl": "mahal",
    "cpt": "cepat", "cpat": "cepat",
    "lmbt": "lambat", "lama": "lama",
    "aman": "aman",
    "rapi": "rapi",
    "sesuai": "sesuai", "ssuai": "sesuai",
    "puas": "puas",
    "kecewa": "kecewa",
    "rusak": "rusak", "rsk": "rusak",
    "cacat": "cacat",
    "baru": "baru",
    "ori": "original", "orisinil": "original",
    "asli": "asli",
    "palsu": "palsu",

    # ── Kata sapaan & salam ───────────────────────────────────
    "mksh": "terima kasih", "makasih": "terima kasih",
    "mks": "terima kasih", "tks": "terima kasih",
    "thx": "terima kasih", "tx": "terima kasih",
    "thanks": "terima kasih", "thank": "terima kasih",
    "oke": "oke", "ok": "oke",
    "sip": "oke",
    "kk": "kakak", "kak": "kakak",
    "mas": "mas", "bang": "abang",
    "min": "admin",

    # ── Ekspresi & filler (dihapus = noise) ──────────────────
    "wkwk": "", "wkwkwk": "", "wkwkwkwk": "",
    "haha": "", "hehe": "", "hihi": "", "huhu": "",
    "ahahaha": "", "hihihi": "",
    "hmm": "", "hm": "", "emmm": "",
    "eh": "", "ih": "", "duh": "",
    "loh": "", "lah": "", "dong": "", "deh": "",
    "nih": "", "sih": "", "aja": "",
    "oot": "",

    # ── Kata promosi / umum di ulasan ────────────────────────
    "recommended": "direkomendasikan",
    "recomended": "direkomendasikan",
    "rekomen": "direkomendasikan",
    "rekomendasikan": "direkomendasikan",
    "worth": "sepadan",
    "worth it": "sepadan",
    "nice": "bagus",
    "good": "bagus",
    "bad": "buruk",
    "best": "terbaik",
    "top": "terbaik",
    "so so": "biasa saja",
    "sos": "biasa saja",

    # ── Singkatan tempat & pengiriman ────────────────────────
    "jkt": "jakarta", "sby": "surabaya",
    "bdg": "bandung", "mlg": "malang",
    "ygy": "yogyakarta", "jogja": "yogyakarta",
    "jne": "jne", "jnt": "j&t", "sicepat": "sicepat",
    "cod": "bayar di tempat",

    # ── Penulisan angka / satuan ──────────────────────────────
    "rb": "ribu", "jt": "juta",
    "cm": "centimeter", "kg": "kilogram",
    "gr": "gram",
}
