

##  BLUEPRINT PERUSAHAAN JAGAMATA
## Master Blueprint Lengkap — Seluruh Divisi
Aplikasi Pengawasan Kesehatan Mata Anak Berbasis AI On-Device
Entitas: JagaMata (Project Akhir — Mobile Application Programming)
Badan usaha (rencana): CV / PT Micro-SaaS — lihat §7 Legal
## Versi: 1.0 · Oktober 2026
Dokumen induk: Blueprint JagaMata Lengkap v2.0 (teknis) — dokumen ini = sisi bisnis, organisasi, dan eksekusi
lintas divisi
Struktur: 10 divisi/bagian + peta dependensi + master roadmap
## RINGKASAN EKSEKUTIF
JagaMata adalah aplikasi Android yang mendeteksi 3 risiko kesehatan mata anak secara real-time — jarak layar <30
cm, ruangan gelap, dan postur menunduk — lalu mengingatkan lewat maskot ramah anak, sementara orang tua
menerima laporan mingguan berbasis data. Seluruh AI berjalan on-device (MediaPipe FaceLandmarker + ML Kit),
tanpa server, tanpa foto yang disimpan.
DimensiAngka Kunci
MasalahMiopia diproyeksikan 49,8% penduduk dunia 2050; Indonesia screen time #1 dunia (6,05 jam/hari)
Celah pasarApple Screen Distance = iOS saja; 77,07% pengguna Indonesia = Android
Model bisnisFreemium langganan (Rp29rb/bln · Rp199rb/thn) + B2B sekolah
Proyeksi pendapatan Y1Rp132,7 jt (kons.) · Rp434,8 jt (median) · Rp1,419 M (optimis)
Gross margin±85–90% — inferensi on-device, biaya server ≈ 0
Unit economics (median)CAC ≈ Rp26 rb · LTV ≈ Rp250–400 rb · LTV:CAC ≈ 10:1
Tim inti5–6 orang (4 divisi inti + 2 pendukung)
FaseFase 0 validasi (6 minggu dev) → Fase 1 peluncuran ID → Fase 2 iOS → Fase 3 SEA
Posisi kompetitif (jujur & teruji riset Okt 2026): kompetitor screen-distance memang ADA (iVisionGuard, Guby,
EyeKeeper, EyeGuard, Plano) — namun tidak ada yang menggabungkan: deteksi 3 sinyal + laporan data 1-menit
yang bisa diekspor ke WhatsApp + harga lokal + paket sekolah + Bahasa Indonesia. Itu diferensiasi JagaMata
## (§1.3).

## BAGIAN 1 — DIVISI STRATEGI & EKSEKUTIF
## 1.1 Visi, Misi, Nilai
VisiMenjadi standar kebiasaan layar sehat anak Indonesia — "Google Screen Time-nya kesehatan mata"
MisiMenjaga jarak aman anak terhadap layar dengan AI murah yang berjalan di HP mana pun, tanpa mengirim satu piksel pun ke
internet
Nilai(1) Privasi by design, (2) Bukti sebelum klaim, (3) Ramah anak, (4) Murah untuk Indonesia
1.2 Posisi Pasar & Analisis Kompetitor (Riset Okt 2026)
A. Kompetitor Langsung (screen-distance / eye-health untuk anak)
KompetitorFokusBahasaHargaKekuatan
Kelemahan untuk pasar
## ID
## Plano (co.plano) —
500K+ unduh, 3,7★
Jarak 30cm + postur +
low light + istirahat 30
mnt
ENGratis (fitur
mata)
Fitur terlengkap;
backing klinik
optometri
## Bukan Bahasa Indonesia;
laporan terbatas; tanpa
paket sekolah di Play ID
iVisionGuardJarak layar saja (+
suara ortu)
7 bhs —
tanpa ID
## Gratis + Plus
langganan
Bebas iklan, mode
hemat baterai
Hanya 1 sinyal; tanpa
laporan ortu berbasis data
GubyJarak layar, maskot
burung hantu, usia 2–8
ENLanggananDesain anak paling
baik; rekap
mingguan via email
Hanya 1 sinyal; tanpa
Bahasa Indonesia; tanpa
ekspor WA
EyeKeeperJarak + timer hadiahENFreemiumDinosaur overlay
menarik
Tanpa laporan ortu
mendalam; tanpa ID
EyeGuardJarak (gentle/strict)ENFreemiumSederhanaFitur paling sedikit

B. Kompetitor Tidak Langsung (parental control — memakai waktu, bukan jarak)
KompetitorHarga tahunan (USD, Okt 2026)Fokus utama
Bark$99/thn (tanpa batas anak/perangkat)Monitoring konten sosial 30+ platform
Qustodio$59,95/thn (5 perangkat) – $104,95 unlimitedScreen time + web filter
Net Nanny$54,99/thn (5 perangkat)Filter web real-time
Norton Family$49,99/thnBudget all-rounder
Aura$120/thnIdentity protection keluarga
Mobicip$35,88–95,88/thnJadwal & filter
Apple Screen TimeGratisBawaan iOS — tidak ada di Android
Google Family LinkGratisBawaan Android — tanpa deteksi jarak/gelap/postur
Insight kunci: seluruh kompetitor global memakai Bahasa Inggris dan harga $50–120/tahun (≈Rp800rb–1,9 juta);
kompetitor gratis (Family Link, Screen Time) tidak deteksi risiko mata sama sekali.
1.3 Enam Pilar Diferensiasi JagaMata (jawaban "beda dari yang lain")
#PilarBuktiTidak dimiliki kompetitor
D1Laporan data 1-menit yang bisa DIEKSPOR (PDF/WA) —
ortu menerima "Selasa malam 3× dekat, 210 lux, 34°"
Blueprint v2 §5.5, §9Plano/Guby = statistik in-app; tanpa
ekspor
D2Paket B2B SEKOLAH — dasbor wali kelas, laporan kelas,
siswa tidak diremehkan (mode siswa)
§2.4Tidak ditemukan di riset Play ID
D3Harga lokal 5–8× lebih murah — Rp199rb/thn vs Bark
$99 (Rp1,58 jt)
§2.2Seluruh kompetitor global
D4Bahasa Indonesia penuh + maskot & copy lokaliVisionGuard tak
punya bahasa ID
Semua kompetitor direct
D53 sinyal + siklus 60 detik + KPI terpublikasi (akurasi diuji
180 sampel ground-truth)
Blueprint v2 §5.7Kompetitor tidak mempublikasikan
akurasi
D6Privasi keras: 0 foto tersimpan, 0 server, sesuai UU PDP§7Juga dimiliki kompetitor — tapi menjadi
syarat kepercayaan ortu ID
Ringkas: kompetitor menjual screen time atau screen distance; JagaMata menjual sistem kebiasaan kesehatan mata
anak berbasis data + laporan ke pihak tepercaya (ortu/sekolah) dalam Bahasa Indonesia.

1.4 Model Keunggulan Berkelanjutan (Moat)
- Dataset ground-truth lokal (180+ sampel foto anak Indonesia, berbagai cahaya) → akurasi yang sulit ditiru
kompetitor asing.
- Jaringan sekolah (B2B) → retensi & distribusi organik via guru/wali kelas.
- Lokalisasi mendalam (copy, budaya "tutup wajah = nasehat", ekosistem WA) → switching cost emosional.
- Harga → skala on-device membuat margin tetap sehat bahkan di Rp199rb.
## BAGIAN 2 — DIVISI BISNIS & MONETISASI
2.1 Ikhtisar Model: Freemium Langganan + B2B
## ┌────────────────────────────────────────────┐
## │            SUMBER PENDAPATAN               │
## ├──────────────┬──────────────┬──────────────┤
## │  1. KONSUMEN │  2. SEKOLAH  │  3. EKOSISTEM│
│  (freemium   │  (B2B tahunan)│ (add-on)    │
│   langganan) │              │              │
## ├──────────────┼──────────────┼──────────────┤
│ Gratis: deteksi│ Paket kelas: │ Aset tematik │
│ jarak dasar   │ dasbor guru  │ maskot,      │
│ Premium:      │ laporan kelas│ tema,        │
│ 3 sinyal +    │ pilot per    │ bundling alat│
│ laporan WA +  │ institusi    │ kesehatan mata│
│ 3 anak        │              │ (afiliasi)   │
## └──────────────┴──────────────┴──────────────┘
Target mix Y1: 85% │ 12% │ 3%

2.2 Struktur Harga (Rupiah — disesuaikan daya beli)
TierHargaIsiTarget
Gratis (Forever)Rp0Deteksi jarak + maskot + 1 anak +
banner kecil
Akuisisi massal; setara fitur
kompetitor gratis
Premium BulananRp29.000/bln3 sinyal penuh + laporan mingguan +
ekspor WA + 3 anak + streak
Uji kelayakan bulanan
Premium Tahunan ⭐Rp199.000/thn
(≈Rp16.600/bln)
Sama, diskon 43%Tulang punggung
pendapatan
Keluarga BesarRp249.000/thn5 anak + 2 profil ortuUpsell
Sekolah DasarRp4.500.000/thn (per 150
siswa)
Dasbor wali kelas + laporan bulanan +
mode siswa
## B2B
## Sekolah
Menengah/Rumah Sakit
Rp12.000.000/thnMulti-kelas + klinik mitra + export massalB2B premium
Banding harga (validasi posisi):
ProdukHarga/tahunSetara IDR*
Bark$99≈Rp1.580.000
Qustodio Complete$104,95≈Rp1.679.000
Net Nanny (5 dev)$54,99≈Rp879.000
Norton Family$49,99≈Rp799.000
JagaMata PremiumRp199.000≈$12,4
(kurs Rp16.000/USD; harga kompetitor dari vendor page Okt 2026)
→ JagaMata = 5–8× lebih murah dari pembanding global, dengan fitur yang lebih relevan untuk ortu Indonesia
(laporan WA, Bahasa ID, sekolah).

2.3 Unit Economics (Skenario Median Y1 — konsisten Blueprint v2)
MetrikNilaiDasar hitung
Unduhan Y150.000Skenario median (benchmark Adapty/RevenueCat: 43% app ID unduh 10rb–
## 100rb)
Konversi berbayar3,8%Benchmark Adapty tier "decent"
Pelanggan berbayar1.90050.000 × 3,8%
ARPU campuran≈Rp228.800Mix tahunan/bulanan (Rp434,8 jt ÷ 1.900)
Pendapatan bruto Y1Rp434,8 juta
Komisi Play (15% tier pertama)−Rp65,2 juta
Pendapatan bersih (net
revenue)
Rp369,6 juta
Biaya infrastruktur
## (server/analytics)
≈Rp0On-device
Biaya pemasaran Y1−Rp45.000.000Konten, ASO, komunitas, uji iklan
Biaya tools, legal, aset, kontingensi−Rp30.000.000§6
Laba sebelum insentif tim Y1≈Rp294,6 juta
Gross margin±85%Net revenue − COGS hampir nol
Rasio kunci:
CAC (biaya akuisisi per pelanggan) = Rp45.000.000 / 1.900  ≈ Rp26.300
LTV konservatif (churn 7%/bln, ARPU thn) ≈ Rp250.000 – Rp400.000
LTV : CAC  ≈ 10 : 1          (target sehat > 3:1) ✅
Payback    < 1 bulan          (Rp26rb vs ARPU bulanan) ✅
CAC payback di bulan pertama langganan tahunan ✅

2.4 Sumber Pendapatan #2: Paket Sekolah (B2B) — kenapa ini pembeda
ElemenRincian
Masalah sekolahProgram "kurangi screen time" sulit dievaluasi tanpa data
Penawaran JagaMataDasbor wali kelas: % siswa berisiko/minggu, laporan PDF bulanan, mode siswa (tanpa pengawasan berlebih)
HargaRp4,5 jt/tahun per 150 siswa (≈Rp30.000/siswa/tahun)
Sales cyclePilot 1 bulan gratis → kontrak tahunan
Distribusi1 sekolah = 150 akun ortu (channel multiplier!)
Target Y15–15 sekolah pilot → Rp22,5–67,5 jt (termasuk dalam skenario median konservatif sebagai bauran 12%)
2.5 Sumber Pendapatan #3: Ekosistem Add-On
Add-OnModelEstimasi Y1
Tema & maskot premium (musim, karakter)IAP sekali beli Rp15.000–
## 25.000
Rp15–30 jt
Afiliasi alat kesehatan mata (kacamata anti-radiasi, lampu belajar ergonomis,
jam tangan anak)
Komisi 5–12% via linkRp10–25 jt
Laporan dokter/orto (PDF format klinis untuk konsultasi)Fitur Premium+Inkremental konversi
White-label untuk klinik optometriLisensi tahunanFase 3 (Rp150
jt+/kontrak)
## 2.6 Roadmap Monetisasi 3 Tahun
TahunFokusPelanggan
## Pendapatan
## (median)
## Catatan
Y1Validasi + langganan konsumen +
pilot sekolah
1.900Rp434,8 jtBreak-even bulan ke-4/5 (biaya
≤Rp75 jt)
Y2Ekspansi iOS + 50 sekolah +
tema/afiliasi
## 6.500 (konsumen) + 50
sekolah
Rp1,7–2,2 MiOS ARPU 2×; SEA 1 negara
## (MY/SG)
Y3SEA (MY, SG, PH) + white-label klinik
+ enterprise
18.000 + 200 sekolahRp4,5–6 MMulai tim berbayar penuh
Uji harga yang direncanakan (sebelum fiksasi): A/B Rp149rb vs Rp199rb vs Rp299rb/thn pada 10% pengguna baru (4
minggu) — berhenti pada titik net revenue per pengunjung tertinggi.

## BAGIAN 3 — DIVISI ENGINEERING (PEMBUATAN UNTUK
## DEVELOPER)
Spesifikasi implementasi lengkap mengacu Blueprint JagaMata v2.0 (arsitektur, kode, KPI). Bagian ini = standar
kerja, struktur tim, sprint, dan definition of done.
3.1 Standar Teknis (WAJIB)
AreaStandar
Bahasa100% Kotlin; Compose UI; coroutines/Flow; tanpa Java kecuali library
Min/Target SDKminSdk 24 (Android 7) · targetSdk 35
ArsitekturClean-ish: detection/, service/, data/ (Room), ui/ (Compose), report/
StateSingle source of truth: Room + StateFlow; tanpa global mutable state
DeteksiMediaPipe FaceLandmarker (LIVE_STREAM) + ML Kit fallback; kamera 5 fps
SiklusMinuteTicker 60 detik (±2 dtk) — in-service coroutine, BUKAN PeriodicWorkRequest
Kualitas kodeKtlint + detekt; coverage unit test ≥70% di detection/ & data/
GitTrunk-based + branch feat/*, fix/*; PR wajib 1 review; conventional commits
CIGitHub Actions: build → test → lint → APK artifact (merge ke main)
DokumentasiREADME per modul + KDoc di fungsi deteksi
3.2 Pemetaan Tim Engineering (4–5 orang)
PeranTanggung JawabOutput
Tech LeadArsitektur, review, risiko MediaPipe, CIKeputusan teknis, PR review
ML/Deteksi EngineerFaceLandmarker, kalibrasi, state machine, akurasiModul detection/ + hasil uji 180 sampel
Android Engineer ACameraX, ForegroundService, MinuteTicker, overlayService & alarm jalan di 5 app
Android Engineer BRoom, dashboard Compose, laporan/export, settingsDashboard + ekspor WA
QA/DevOps (paruh waktu)Ground truth, pipeline metrics.py, perangkat ujiTabel KPI, bug tracker

3.3 Sprint Plan 6 Minggu (2 minggu/sprint, 3 sprint)
SprintMingguDeliverableDefinition of Done
S1M1–M2Kalibrasi jarak + state machine + fallback ML KitK5 latensi ≤1,5 dtk; status stabil 10 mnt; unit test
merah→hijau
S2M3–M4Overlay maskot + 3 sinyal + wizard izinK10 overlay di 5 app; 3 peringatan bersama; wizard
lengkap
S3M5–M6MinuteTicker + DB + dashboard + laporan +
seluruh KPI
K4: 60 mnt = 60 baris; 10/10 KPI hijau; video demo
Aturan sprint: demo internal tiap Jumat · bug blocker = tunda rilis · perubahan scope hanya via Product Owner
(dosen/ketua tim) · buffer 2 hari di Sprint 3 untuk perbaikan KPI gagal.
## 3.4 Struktur Repository & Pipeline Data Uji
jagamata/                        (GitHub — private)
├── app/                         (Android module utama)
├── ml/                          (evaluasi: metrics.py, ground_truth/*.csv)
│   ├── ground_truth/photos/     (180 foto + labels.csv)
│   └── reports/                 (hasil KPI per minggu → lampiran laporan)
├── docs/                        (arsitektur, keputusan teknis ADR)
## ├── .github/workflows/ci.yml     (build+test+lint)
└── README.md                    (setup, cara reproduce uji)
Pipeline data (offline — tanpa PII):
foto uji → app (mode kalibrasi) → CSV prediksi {id, jarak_pred, pitch, lux}
→ metrics.py (Python) → {sensitivitas, FPR, MAE} → tabel laporan
Aturan emas: tidak pernah mengirim foto ke repository/cloud — hanya angka.

## 3.5 Risiko Teknis & Rencana Mitigasi Engineering
RisikoProb.DampakMitigasiCadangan
Akurasi jarak gagal ≥90% di HP
murah
SedangKPI K1 gagalKalibrasi per-device + smoothing EMA
+ uji 3 HP
Naikkan ambang + laporkan
MAE jujur
OEM kill service (MIUI/ColorOS)TinggiFitur matiWizard auto-start + whitelist bateraiDeteksi service mati →
banner peringatan
Overlay permission ditolakSedangAlarm tak
tampil
Wizard + video panduanFallback notifikasi penuh
Google rilis fitur sejenisRendahRedundanDiferensiasi D1–D4
## (laporan+sekolah+lokal)
Pivot ke B2B sekolah
Play Store tolak (kebijakan
anak/kesehatan)
RendahRilis tertahanDisclaimer, Data Safety jujur, tanpa
klaim medis
Revisi listing 7 hari
## BAGIAN 4 — DIVISI PRODUK & DESAIN
## 4.1 Prinsip Produk
- Anak bukan target "yang diawasi" — dia pemain yang dibantu (maskot, bahasa positif, tanpa label "nakal").
- Ortu = pembayar — semua insight harus terbaca dalam 5 detik di dashboard.
- Sekali set-up, jalan sendiri — onboarding ≤3 menit, kalibrasi 1×, lalu app invisible sampai ada pelanggaran.
4.2 Peta Fitur (Prioritas MoSCoW)
PrioritasFiturStatus
MustDeteksi jarak 30cm, gelap 150lux, postur 30°; alarm maskot; kalibrasi; siklus 60 detik; dashboard
ortu; laporan mingguan export
## Fase 1 (dev 6
minggu)
ShouldStreak/badge gamifikasi; Mode Ringan hemat baterai; ekspor WhatsApp; multi-anak (3)Fase 1b (bulan 2–
## 3)
CouldTema maskot premium; laporan dokter PDF; widget dashboard; wear OS/watchFase 2
## Won't
## (Y1)
Telemedicine, diagnosis, data ke cloud, iOS (masuk Fase 2)Tunda

4.3 Standar Desain & UX
AspekStandar
Desain anakMaskot Bimo si burung hantu (mata besar = kesehatan mata), 3 state: senang/kaget/mengingatkan; palet pastel +
kontras WCAG AA
BahasaIndonesia ramah anak ("Ayo mundur dikit, Bimo jaga matamu!") — TANPA kata menghakimi
AksesibilitasFont ≥14sp; dukungan TalkBack di dashboard; mode buta warna (hijau/merah + ikon)
KecepatanDashboard <200 ms; cold start <1,5 dtk
## Privasi-
terlihat
Badge " 0 foto keluar dari HP" di onboarding
4.4 Riset Pengguna Praluncur (wajib sebelum Fase 1)
AktivitasSampelOutput
Wawancara ortu (pain point, kebiasaan bayar)8–10 ortu anak 5–13 thValidasi harga Rp199rb & kepercayaan privasi
Uji kegunaan onboarding5 ortuWaktu set-up, titik bingung
Uji reaksi anak pada alarm5 anakApakah maskot menakutkan? (iterasi copy)
## BAGIAN 5 — DIVISI PEMASARAN & PERTUMBUHAN
## (GROWTH)
5.1 Go-to-Market — 3 Fase
FASE 1 (Bulan 1–3): VALIDASI & ORGANIK ID
ASO + komunitas ortu + parenting influencer mikro + uji iklan kecil
## ↓
FASE 2 (Bulan 4–9): SKALA KONSUMEN + PILOT SEKOLAH
Kampanye "Mata Sehat Sekolah" + kemitraan klinik optometri + konten kesehatan
## ↓
FASE 3 (Bulan 10–18): iOS + EKSPANSI SEA
App Store (ARPU 2×) → Malaysia/Singapura (pasar 45% iOS)

5.2 Kanal & Taktik (dengan KPI)
KanalTaktikKPIAnggaran Y1
ASO (App Store
## Optimization)
Kata kunci: "jarak layar anak", "miopia", "screen
distance", "kesehatan mata anak"
Konversi listing ≥35%Rp0 (organik)
Konten edukasiTikTok/Instagram/YouTube Shorts: "5 tanda mata anak
lelah"; kolaborasi dr. mata
100rb views/bln; 500
unduh/bln
Rp15 jt (produksi +
honor mikro)
Komunitas ortuGrup WhatsApp, forum parenting, mom/dad
influencer (5–50rb followers)
30% unduhan dari
referral
Rp12 jt
SekolahRoadshow "Kelas Mata Sehat", pilot gratis 1 bulan15 sekolah pilotRp10 jt (materi +
transport)
Klinik optometriPoster QR + afiliasi20 klinik mitraRp5 jt (materi)
Iklan berbayarUji Meta/Google Ads 4 mingguCAC <Rp30.000/unduhRp3 jt (uang uji)
Total anggaran pemasaran Y1: Rp45.000.000 (sesuai unit economics §2.3).
## 5.3 Fungel Pertumbuhan
Impresi listing (ASO/konten) → Klik install → Aktivasi (kalibrasi selesai ≥70%) →
Bayar (3,8%) → Retensi 90 hari (≥40%) → Renew tahunan (≥55%) → Referral ke ortu lain
Metrik aktivasi kunci: % pengguna yang menyelesaikan kalibrasi — tanpa ini, app tidak berguna. Target: ≥70% dari
pembuka pertama.
5.4 Positioning Pesan (copy utama)
"Bukan berapa lama anak main HP — tapi seberapa dekat. JagaMata menjaga jarak aman mata anak, lalu
melaporkannya ke Anda setiap minggu. 100% offline, tanpa foto."
AudiensPesan
Ortu"Jarak aman 30 cm — Bimo mengingatkan, Anda melihat datanya"
Guru/wali kelas"Kelas Anda: berapa siswa berisiko minggu ini? Satu laporan PDF"
Dokter/optometris"Alat pendukung kebiasaan — data 60 detik siap dibawa ke konsultasi"

## BAGIAN 6 — DIVISI OPERASIONAL & KEUANGAN
6.1 Struktur Biaya (Cost Structure)
KategoriItemY1 (Rp)Sifat
PemasaranKonten, influencer, sekolah, iklan uji45.000.000Variabel
Tools & LayananPlay Console (USD 25), GitHub, desain, domain, analytics8.000.000Tetap
LegalNama usaha, kontrak sekolah, UU PDP, disclaimer12.000.000Sekali (Fase 1)
Perangkat uji2–3 HP bekas uji + penggaris + perlengkapan6.000.000Sekali
Kontingensi15% dari total10.350.000Cadangan
Insentif timInsentif/bonus hasil (bukan gaji penuh — tim mahasiswa)15.000.000Kinerja
TOTAL BIAYA Y1≈Rp96.350.000
6.2 Proyeksi Laba–Rugi 3 Tahun (skenario median)
## Y1Y2Y3
Pendapatan bruto434,8 jt1,95 M5,2 M
Komisi toko (15%)(65,2 jt)(292,5 jt)(780 jt)
Pendapatan bersih369,6 jt1,657 M4,42 M
Biaya operasi(96,4 jt)(420 jt)(1,15 M)
Laba sebelum pajak≈273 jt≈1,24 M≈3,27 M
Margin bersih63%63%63%
Tim berbayar penuh0 (mahasiswa)4–6 orang10–14 orang
Catatan Y2/Y3: asumsi pertumbuhan konservatif 4,5×/tahun (di bawah rata-rata kategori parental control 10–12%/tahun
dikombinasi akuisisi baru) — angka akan direvisi per kuartal berdasarkan data aktual.
## 6.3 Arus Kas & Titik Impas
Biaya awal ≤Rp75 jt (tanpa gaji penuh) → tercakup pendapatan bulan ke-4 (konversi 3,8% mulai berjalan).
Break-even: bulan ke-4 sampai ke-5 Y1 (kumulatif pendapatan > kumulatif biaya).
Runway aman: jangan rekrut gaji penuh sebelum net revenue 3 bulan berturut ≥Rp60 jt/bulan.

6.4 Indikator Kesehatan Bisnis (Dashboard Direksi — bulanan)
IndikatorTarget hijauKuningMerah
MRR+ARR tumbuh MoM≥8%3–7%<3%
Churn bulanan<7%7–10%>10%
Konversi freemium≥3,8%2,9–3,8%<2,9%
CAC<Rp30.000Rp30–50rb>Rp50rb
Aktivasi kalibrasi≥70%50–70%<50%
## Rating Play Store≥4,34,0–4,3<4,0
## BAGIAN 7 — DIVISI LEGAL & KEPATUHAN
## 7.1 Badan Usaha & Kepemilikan
OpsiCocok untukCatatan
CV (fase awal)Kontrak sekolah, tagihan B2BModal kecil, cepat (±2 minggu + NIB OSS)
PT peroranganSkala naik, investorPelaporan lebih rapi
Fase mahasiswaValidasi duluBoleh start perorangan, naikkan saat kontrak B2B pertama
Rekomendasi: validasi dengan perorangan (bulan 1–3) → CV setelah kontrak sekolah pertama (untuk invoicing
resmi).
7.2 Kepatuhan Privasi (UU PDP + Kebijakan Store)
KewajibanImplementasi
Persetujuan eksplisitCheckbox onboarding: izin kamera + kebijakan privasi (tautan URL aktif)
Transparansi dataHalaman "Data & Privasi": 0 foto, 0 server, semua di HP — kutipan per item
Hak hapus dataTombol "Hapus semua data" di pengaturan (hapus Room + DataStore)
Data Safety form (Play)Deklarasi jujur: no data collected; kamera diproses lokal
Kebijakan konten anakTidak ada iklan pihak ketiga di tier anak; tanpa pelacakan lintas-app
PenyimpananSnapshot 30 hari otomatis terhapus (prune)

## 7.3 Kepatuhan Toko Aplikasi
RisikoMitigation
Klaim kesehatan berlebihan (Guideline
Play/Apple 5.1)
Copy "pengingat kebiasaan"; disclaimer tegas: bukan alat diagnosis/pencegah miopia
Review Apple 23% ditolak (Guideline 2.1)Aplikasi final sebelum submit; demo akun; buffer 1–2 minggu
Komisi & pajakPlay Billing 15% (tier 1); setor pajak (PPh final 0,5% PPN?) — akuntan saat revenue >
Rp4,8 M/tahun (UMKM)
## 7.4 Lisensi Komponen
KomponenLisensiKewajiban
MediaPipe (Google)Apache 2.0Cantumkan notice lisensi
ML Kit (Google)Proprietary SDK (gratis)Patuhi Google Mobile SDK ToS; boleh offline
Kotlin/Compose/Room/CameraXApache 2.0Notice
Font ilustrasi/asetBeli lisensi komersial (mis. Freepik Pro/Flaticon)Simpan bukti pembelian
7.5 Kontrak B2B Sekolah (kerangka isi)
- Ruang lingkup (dasbor, jumlah siswa, laporan bulanan) · 2. Durasi & perpanjangan otomatis · 3. Data: ortu/siswa =
milik sekolah, JagaMata hanya memproses lokal · 4. Dukungan (SLA respons 1×24 jam) · 5. Pembayaran tahunan di
muka · 6. Uji coba 1 bulan gratis tanpa kewajiban · 7. Pengakhiran & ekspor data.
## BAGIAN 8 — DIVISI RISET & QUALITY (QA)
8.1 Metodologi Ground Truth (inti kredibilitas)
- Siapkan 6 titik jarak: 20 │ 25 │ 30 │ 35 │ 40 │ 50 cm (penggaris logam)
- 30 sampel per titik (3 kondisi cahaya × 10) → 180 foto + labels.csv
- Aplikasi mode kalibrasi → prediksi → metrics.py
- Metrik: Sensitivitas <30cm ≥90% · FPR ≥35cm ≤10% · MAE ≤6 cm
- Uji silang: 3 HP berbeda merek/versi → selisih prediksi <15%

8.2 Papan KPI (harus ijo sebelum rilis) — 10 KPI
(Sama dengan Blueprint v2 §5.7 — K1...K10; K4 = siklus 1 menit ±2 detik.)
#KPITargetPemilik
K1Sensitivitas jarak≥90%ML Engineer + QA
K2False positive≤10%ML Engineer + QA
K3MAE jarak≤6 cmML Engineer
K4Ketepatan siklus 60 detik60 baris/60 mnt, ±2 dtkAndroid Engineer A
K5Latensi alarm≤1,5 dtkAndroid Engineer A
K6FPS HP RAM 3GB≥5 fpsTech Lead
K7Baterai/jam (Normal)≤8%Android Engineer A
K8Offline penuhLulusQA
K9Privasi (0 foto)Lulus auditTech Lead
K10Overlay 5 appLulusAndroid Engineer A
## 8.3 Lapisan Pengujian
LapisanCakupanTool
UnitRumus jarak, state machine, aggregator, Room DAOJUnit + MockK; coverage ≥70% modul inti
InstrumentasiKalibrasi flow, overlay lifecycleEspresso/Compose test
E2E manual10 KPI + 5 app overlay + 3 HPChecklist §8.2
RegresiSetiap rilis: smoke test 15 menitChecklist tetap
A/B monetisasiUji 3 titik harga 4 mingguPlay Console A/B
Aturan: tidak ada rilis dengan KPI merah; hasil uji disimpan di ml/reports/ sebagai lampiran laporan akhir.
## BAGIAN 9 — STRUKTUR ORGANISASI & KPI LINTAS DIVISI

9.1 Peta Organisasi (Tim Proyek 5–6 orang)
## ┌──────────────────────┐
│  KETUA / PRODUCT OWNER│  ← arah bisnis, hubungan dosen/sekolah
## └──────────┬───────────┘
## ┌──────────┬────────┴───────┬──────────────┬──────────────┐
## ┌────▼────┐ ┌───▼────┐   ┌───────▼──────┐ ┌─────▼─────┐ ┌──────▼──────┐
## │ENGINEERING│ │ PRODUCT│   │  PERTUMBUHAN  │ │  QA &     │ │ OPERASIONAL │
│ (3–4 org) │ │& DESAIN│   │  & PEMASARAN │ │  RISET    │ │ & KEUANGAN  │
│ Tech Lead │ │ (1 org)│   │   (1 org)    │ │ (1 org,   │ │ (1 org,     │
│ ML Eng    │ │        │   │              │ │  paruh)   │ │  paruh)     │
## │ Android×2 │ │        │   │              │ │           │ │             │
## └────┬────┘ └───┬────┘   └───────┬──────┘ └─────┬─────┘ └──────┬──────┘
## └──────────┴────────┬───────┴──────────────┴──────────────┘
## ▼
DOKUMEN INDUK: Blueprint JagaMata v2.0 (teknis)
+ Blueprint Perusahaan ini (bisnis)
9.2 KPI per Divisi (dipantau bulanan)
DivisiKPI UtamaTarget
Engineering10/10 KPI teknis · sprint on-time · bug blocker100% · ≥90% · 0 di rilis
Produk & DesainAktivasi kalibrasi · rating · waktu onboarding≥70% · ≥4,3★ · ≤3 mnt
PertumbuhanUnduhan/bln · konversi listing · CAC≥4.000 · ≥35% · <Rp30rb
QA & RisetKelengkapan 180 sampel · laporan mingguan KPI100% · tepat waktu
Operasional & KeuanganBiaya aktual vs anggaran · MRR · break-even≤100% · tumbuh · bulan ≤5
9.3 RACI Singkat (keputusan besar)
KeputusanRACI
Ubah hargaProduct OwnerKetuaGrowth, KeuanganSemua divisi
Ganti ambang deteksi (30→35 cm)ML EngTech LeadQA, ProdukDosen (pembimbing)
Rilis ke Play StoreTech LeadKetuaQA, LegalGrowth
Kontrak sekolahGrowthKetuaLegal, KeuanganEngineering

## BAGIAN 10 — MASTER ROADMAP (INTEGRASI SELURUH
## DIVISI)
10.1 Fase 0 — Persiapan (Hari 1–3)
Setup Android Studio, spike MediaPipe (H2 — titik risiko tertinggi), device uji, GitHub.
10.2 Fase 1 — Validasi Teknis (Minggu 1–6) — lihat Blueprint v2 §16
MingguEngineeringQA & RisetProdukGrowthOperasional
M1Jarak + kalibrasiMulai 60 foto ground
truth
—Riset hargaSiapkan tools
M2State machine + ML KitUji latensi K5Wawancara ortu
## #1
Riset ASO
keywords
## —
M3Overlay maskotUji overlay 5 appUji reaksi anakPra-konten (20
post)
## —
M43 sinyalVideo postur 12Onboarding
polish
Siap landingDraft kontrak
M5MinuteTicker + DB +
dashboard
Uji siklus 60 mntLaporan UXPra-launch
waitlist
Draft kebijakan
privasi
M6Fix KPI + polish10/10 KPI + 180 fotoFinalASO final + 5
konten
Siapkan CV (jika
kontrak)
✅ Milestone Fase 1 (akhir M6): aplikasi lulus 10 KPI + 180 sampel + demo video — GO/NO-GO peluncuran.
10.3 Fase 2 — Peluncuran (Bulan 2–6)
BulanAktivitasTarget
B2–B3Soft launch 100 ortu (komunitas WA) → bug & harga100 pengguna, 3,8% bayar
B4Peluncuran publik Play Store + ASO + 30 konten5.000 unduhan
B5Pilot 5 sekolah + 10 klinik mitra5 kontrak pilot
B6Evaluasi A/B harga + iterasiARR ≥ Rp250 jt

10.4 Fase 3 — Skala (Bulan 7–18)
B7–B9: iOS App Store (ARPU 2×) · 25 sekolah · afiliasi alat kesehatan
B10–B12: SEA (Malaysia — 55% Android/45% iOS → dua versi) · white-label klinik
Tahun 2: tim berbayar 4–6 orang · MRR ≥ Rp140 jt · putaran pendanaan mini (opsional)
10.5 Peta Dependensi Antar-Divisi
Engineering (M1–M6) ──► QA (KPI 10/10) ──► GO peluncuran
## │                                      │
├──► Produk (onboarding, laporan) ─────┤
## │                                      ▼
└──► Growth (konten, ASO, sekolah) ◄── Legal (privasi, disclaimer)
## │
## ▼
Operasional (harga, uang, kontrak)
Ketergantungan kritis: Growth TIDAK boleh menaikkan anggaran iklan sebelum QA lolos K1–K4 (akurasi & siklus) —
iklan di aplikasi yang false-alarm = ulasan 1★ permanen.
## PENUTUP — DEKLARASI PEMILIK USAHA
Blueprint ini menyatukan tiga lapis project JagaMata:
- Teknis — Blueprint JagaMata Lengkap v2.0 (arsitektur, kode, siklus 60 detik, 10 KPI);
- Bisnis & Organisasi — dokumen ini (diferensiasi D1–D6, unit economics, 9 divisi);
- Akademik — BAB 6 Platform Perangkat Bergerak × JagaMata.
Keputusan yang sudah dikunci: Android-First · freemium Rp199rb/thn · siklus 60 detik · B2B sekolah · 0 foto tersimpan
· 6 minggu ke GO/NO-GO.
Keputusan berikutnya (butuh persetujuan): badan usaha (CV saat kontrak B2B pertama) · titik harga final setelah A/B
· keputusan iOS (setelah ARR Y1 terbukti).
## REFERENSI
- StatCounter Global Stats — Mobile OS Market Share: Indonesia, Asia (2026).
- Sensor Tower — State of Mobile 2026 (non-game IAP USD 85,6 M).
- RevenueCat — State of Subscription Apps 2025/2026.
- Adapty — Subscription App Benchmarks 2025/2026 (konversi 1,6% / 2,9% / 6,2%).

- WiseGuy (2025) — Parental Control App market USD 2,92 M → 6,5 M (2035).
- HiWave, CyberNews, Cyberpresso, RaisingBytes, Memeburn (Okt 2026) — perbandingan harga Bark/Qustodio/Net
Nanny/Aura/Norton/Mobicip.
- Google Play Store — listing Plano (co.plano), iVisionGuard, Guby, EyeKeeper, EyeGuard (diakses Okt 2026).
- Apple — App Store Transparency Report 2025 (9.100.620 review; 23,00% ditolak).
- Republik Indonesia — UU No. 27 Tahun 2022 tentang Pelindungan Data Pribadi.
- Holden dkk. (2016) — Global Prevalence of Myopia, Ophthalmology 123(5).
- data.ai (2024) — Indonesia screen time #1 dunia (6,05 jam/hari).
- Blueprint JagaMata Lengkap v2.0 (internal, 2026) — spesifikasi teknis & KPI.
- BAB 6: Platform Perangkat Bergerak (internal, 2026) — kajian Android/iOS.
Blueprint Perusahaan JagaMata v1.0 · Dokumen lintas divisi · Disusun untuk eksekusi tim & presentasi ke
dosen/pembimbing