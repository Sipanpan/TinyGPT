import urllib.request
import json
import re

topics = ["Tata_Surya", "Matahari", "Merkurius", "Venus", "Bumi", "Mars", "Jupiter", "Saturnus", "Uranus", "Neptunus", "Bulan", "Komet", "Asteroid"]
corpus = ""
word_count = 0

for topic in topics:
    url = f"https://id.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json&titles={topic}"
    try:
        req = urllib.request.Request(
            url, 
            data=None, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        res = urllib.request.urlopen(req).read()
        data = json.loads(res)
        pages = data["query"]["pages"]
        for page_id in pages:
            extract = pages[page_id].get("extract", "")
            corpus += extract + "\n\n"
            word_count += len(extract.split())
            
        if word_count >= 2500:
            print("Target jumlah kata tercapai, menghentikan unduhan...")
            break
            
    except Exception as e:
        print(f"Failed {topic}: {e}")

# [DATA CLEANING] Hapus simbol heading seperti == Asal-usul ==
corpus = re.sub(r'={2,}[^=]+={2,}', '', corpus)

# Hapus whitespace berlebih akibat penghapusan heading
corpus = re.sub(r'\s+', ' ', corpus).strip()

# Memotong persis menjadi maksimal 2500 kata agar presisi
words = corpus.split()
if len(words) > 2500:
    words = words[:2500]
    word_count = 2500

# Gabungkan kata-kata kembali
corpus = " ".join(words)

# [DATA CLEANING] Pemisahan kalimat: ganti titik spasi menjadi baris baru.
# Regex ini mencegah titik yang didahului SATU huruf kapital (seperti C. atau R.) agar tidak terpotong.
corpus = re.sub(r'(?<!\b[A-Z])\. ', '.\n', corpus)

with open("corpus.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print(f"Corpus saved with approximately {word_count} words.")
