// Lõi "điện thoại làm kính": gửi khung camera, nhận câu + WAV, báo động mất kết nối.
// Dùng chung cho thu_kinh.html (trang thử) và app.html (app VisionAid). Trang phải có các phần tử id:
// cham, trangthai, tre, cau, canh, khongcb, nut, xem, battieng, cam, cv và nút chọn input[name=cd].
'use strict';
// Giao thức: docs/superpowers/specs/2026-10-24-thu-kinh-truc-tiep-design.md
// Mở từ biểu tượng ngoài màn hình chính (app đã cài): địa chỉ không có mã -> dùng mã của lần quét QR gần nhất.
// Mã đổi mỗi lần bật máy tính: mã cũ bị từ chối (1008) -> trang báo "Sai mã – quét lại mã QR".
function thamSo() {
  const KHOA = 'visionaid_tham_so';
  try {
    if (new URLSearchParams(location.search).get('k')) localStorage.setItem(KHOA, location.search);
    else return new URLSearchParams(localStorage.getItem(KHOA) || '');
  } catch (e) {}
  return new URLSearchParams(location.search);
}
const THAM = thamSo(), MA = THAM.get('k') || '';
// may=<máy tính>: app mở từ GitHub Pages (địa chỉ cố định), máy tính chạy AI ở chỗ khác (đường hầm / IP nội bộ).
// Không có: máy tính chính là nơi phục vụ trang này.
const MAY = THAM.get('may') || '';
// khung=43: cắt giữa 4:3 640x480 như camera kính ESP32 (FRAMESIZE_VGA); fps: số khung/giây gửi đi (ESP32 3–8)
const KHUNG_43 = THAM.get('khung') === '43', CW = 640, CH = KHUNG_43 ? 480 : 360;
const FPS = Math.min(10, Math.max(1, Number(THAM.get('fps')) || 10));
const TRE_BO_MS = 1500;      // câu tới trễ hơn mức này (so với lúc chụp) -> bỏ
const MAT_KET_NOI_MS = 2500; // không có phản hồi lâu hơn -> tự báo động (đường hầm Cloudflare khựng tới ~1 s là thường: đo 2026-10-28)
// Giới hạn số khung trên đường: gửi không giới hạn thì mạng chậm làm khung xếp hàng, độ trễ phình tới vài giây
// (đo thật 25/10: 0,3 s -> 5,6 s). Chỉ 1 khung thì 4G (đi-về ~0,5 s) chỉ được ~2 khung/giây, mà module cần 3 khung
// liền mới báo -> chậm ~1,5 s. Mạng chậm do đi-về: 2 khung gấp đôi khung/giây, trễ không đổi. Mạng chậm do thiếu
// băng thông: khung thứ 2 chỉ xếp hàng sau khung đầu (trễ x3, không thêm khung) -> thấy trễ vượt mức thấp nhất vừa đo
// hơn hẳn thì về 1 khung, THU_LAI_MS sau mới thử lại 2 (thử nghiệm bóp mạng 25/10, xem progress.md).
const SO_KHUNG_TOI_DA = 2;
const THU_LAI_MS = 10000;
const CHO_TOI_DA_MS = 2000;  // lưới an toàn: phản hồi mất hẳn thì sau mức này vẫn gửi khung mới
const $ = id => document.getElementById(id);

let chay = false, cheDo = 'truc_tiep', saiMa = false;
let ws = null, ac = null, luong = null, wakeLock = null, ghi = null, manh = [];
let soKhungTrenDuong = 1, nghenLuc = 0, treGanDay = [];   // [lúc, ms] phản hồi 10 s gần nhất
let idKhung = 0, idDaXong = 0, guiLuc = 0, ketNoiLuc = 0, phanHoiCuoi = 0, baoDongLuc = 0, dangMat = false;
let jsonCho = null;                   // JSON có câu, đang chờ tin wav đi kèm
let cauDangHoi = "";                  // câu nút Hỏi đang chờ trả lời (giữ dòng "Đang hỏi" trên màn hình)
let dangPhat = null;                  // {nguon, id, t_chup, cau, audio}
let cauCho = null;                    // câu thường tới lúc đang nói: phát sau nếu còn mới
const am = {};                        // âm báo động đã giải mã

function trangThai(chu, mau) { $('trangthai').textContent = chu; $('cham').style.background = mau; }
function canh(chu) { $('canh').textContent = chu || ''; }

// Âm câm lặp vô hạn qua <audio>: đưa iOS vào chế độ phát nhạc để Web Audio kêu cả khi gạt im lặng
function amCam() {
  const n = 1600, b = new ArrayBuffer(44 + n * 2), v = new DataView(b);
  const s = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  s(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); s(8, 'WAVEfmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, 16000, true); v.setUint32(28, 32000, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true); s(36, 'data'); v.setUint32(40, n * 2, true);
  return URL.createObjectURL(new Blob([b], { type: 'audio/wav' }));
}

async function napAm(ten) {
  const r = await fetch('static/' + ten + '.wav');   // tương đối: chạy cả ở '/' lẫn '/VisionAid/' (Pages)
  am[ten] = await ac.decodeAudioData(await r.arrayBuffer());
}

function phatBuffer(buf, onXong, sau = 0) {
  const nguon = ac.createBufferSource();
  nguon.buffer = buf;
  nguon.connect(ac.destination);
  nguon.onended = onXong || null;
  nguon.start(ac.currentTime + sau);
  return nguon;
}

function baoDong(ten) {
  if (!am[ten]) return;
  if (dangPhat) {                    // báo động cắt câu đang nói: báo đúng là bị cắt, bỏ câu chờ (đã cũ)
    const cu = dangPhat; dangPhat = null;
    try { cu.nguon.onended = null; cu.nguon.stop(); } catch (e) {}
    guiPhanHoi('bi_cat', cu.id, Date.now() - cu.t_chup);
  }
  if (cauCho) { guiPhanHoi('thay', cauCho.id, Date.now() - cauCho.t_chup); cauCho = null; }
  // bíp ngắn trước câu báo động
  const o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = 880; g.gain.value = 0.3; o.connect(g); g.connect(ac.destination);
  o.start(); o.stop(ac.currentTime + 0.15);
  phatBuffer(am[ten], null, 0.2);   // hẹn trên đồng hồ âm thanh: Safari đóng băng setTimeout khi trang bị ẩn
}

function bip(hz, giay) {
  if (!ac || ac.state !== 'running') return;
  const o = ac.createOscillator(), g = ac.createGain();
  o.frequency.value = hz; g.gain.value = 0.25; o.connect(g); g.connect(ac.destination);
  o.start(); o.stop(ac.currentTime + giay);
}

// Câu báo của chính app (lỗi hỏi, chưa nghe rõ...): giọng tiếng Việt có sẵn trên điện thoại, không cần máy tính
function noiTaiCho(chu) {
  try {
    const u = new SpeechSynthesisUtterance(chu); u.lang = 'vi-VN'; u.rate = 1.1;
    speechSynthesis.cancel(); speechSynthesis.speak(u);
  } catch (e) {}
}

// Nút Hỏi: gửi câu hỏi (chữ) lên máy tính; máy tính ghép với khung mới nhất, hỏi Gemini, trả câu + WAV
function guiCauHoi(cau) {
  cau = (cau || '').trim();
  let loi = '';
  if (!cau) loi = 'Chưa nghe rõ câu hỏi.';
  else if (!chay || cheDo !== 'truc_tiep' || !ws || ws.readyState !== 1) loi = 'Chưa nối máy tính, chưa hỏi được.';
  if (loi) { canh(loi); noiTaiCho(loi); return false; }
  ws.send(JSON.stringify({ lenh: 'hoi', cau }));
  canh('Đang gửi câu hỏi: ' + cau);
  return true;
}

function guiPhanHoi(phan_hoi, id, tre_ms) {
  if (ws && ws.readyState === 1) ws.send(JSON.stringify({ phan_hoi, id, tre_ms: Math.round(tre_ms) }));
}

async function nhanWav(buf) {
  const j = jsonCho; jsonCho = null;
  if (!j) return;
  const tre = Date.now() - j.t_chup, hoi = j.hoi === 'tra_loi';
  // câu cũ: vật có thể đã qua. Câu trả lời nút Hỏi không cũ theo nghĩa đó (Gemini mất 3–5 s là thường) -> không bỏ
  if (tre > TRE_BO_MS && !hoi) { guiPhanHoi('bo', j.id, tre); return; }
  let audio;
  try { audio = await ac.decodeAudioData(buf); }
  catch (e) { guiPhanHoi('bo', j.id, Date.now() - j.t_chup); return; }   // âm hỏng: bỏ, máy được nói lại
  const p = { id: j.id, t_chup: j.t_chup, cau: j.cau, audio, hoi };
  if (dangPhat) {
    // đang nói: câu thường, hoặc "Dừng lại" khi đang nói "Dừng lại" -> giữ 1 câu chờ (mới nhất) phát sau
    if (!p.cau.startsWith('Dừng lại') || dangPhat.cau.startsWith('Dừng lại')) {
      // câu trả lời đang chờ không bị câu dẫn đường thường thay chỗ (người dùng đã chủ ý hỏi); "Dừng lại" vẫn cắt
      if (cauCho && cauCho.hoi && !p.hoi) { guiPhanHoi('thay', p.id, Date.now() - p.t_chup); return; }
      if (cauCho) guiPhanHoi('thay', cauCho.id, Date.now() - cauCho.t_chup);
      cauCho = p; return;
    }
    const cu = dangPhat; dangPhat = null;                          // "Dừng lại" cắt ngang câu đang nói
    try { cu.nguon.onended = null; cu.nguon.stop(); } catch (e) {}
    guiPhanHoi('bi_cat', cu.id, Date.now() - cu.t_chup);
  }
  phat(p);
}

function phat(p) {
  const tre = Date.now() - p.t_chup;
  p.nguon = phatBuffer(p.audio, () => {
    if (dangPhat !== p) return;
    dangPhat = null;
    guiPhanHoi('da_phat', p.id, tre);
    const c = cauCho; cauCho = null;
    if (c) (Date.now() - c.t_chup > TRE_BO_MS && !c.hoi ? guiPhanHoi('bo', c.id, Date.now() - c.t_chup) : phat(c));
  });
  dangPhat = p;
  $('cau').textContent = p.cau;
}

function capNhatSoKhung(tre) {
  const now = Date.now();
  treGanDay.push([now, tre]);
  while (now - treGanDay[0][0] > 10000) treGanDay.shift();
  const thapNhat = Math.min(...treGanDay.map(x => x[1]));
  if (tre > 1.5 * thapNhat + 100) { soKhungTrenDuong = 1; nghenLuc = now; }   // trễ thêm ~1 lần gửi khung: đang xếp hàng
  else if (now - nghenLuc > THU_LAI_MS) soKhungTrenDuong = SO_KHUNG_TOI_DA;
}

function nhanJson(j) {
  // tin hỏi đáp không gắn với khung nào (không có id): xử lý riêng, không đụng nhịp gửi khung
  if (j.hoi === 'dang_hoi') { cauDangHoi = j.cau_hoi; canh('Đang hỏi: ' + j.cau_hoi); bip(660, 0.1); return; }
  if (j.hoi === 'loi') { cauDangHoi = ''; canh(j.loi_hoi); noiTaiCho(j.loi_hoi); return; }
  if (j.hoi === 'tra_loi') cauDangHoi = '';
  idDaXong = Math.max(idDaXong, j.id);   // máy chủ xử lý theo thứ tự: khung cũ hơn bị bỏ qua cũng coi như xong
  if (j.loi) { canh('Máy tính lỗi: ' + j.loi); return; }   // lỗi liên tục > 1 s -> canhGac báo động như mất kết nối
  phanHoiCuoi = Date.now();
  if (dangMat) { dangMat = false; trangThai('Đã kết nối', 'var(--xanh)'); if (am.da_ket_noi) phatBuffer(am.da_ket_noi); }
  const tre = Date.now() - j.t_chup;
  capNhatSoKhung(tre);
  $('tre').innerHTML = tre + ' <small>ms (xử lý ' + j.ms_xu_ly + ')</small>';
  $('tre').style.color = tre < 300 ? 'var(--xanh)' : tre < 600 ? 'var(--vang)' : 'var(--do)';
  canh(cauDangHoi ? 'Đang hỏi: ' + cauDangHoi : '');   // giữ dòng "Đang hỏi" tới khi có câu trả lời
  if (j.cau) jsonCho = j;
}

function moKetNoi() {
  const u = (MAY ? 'wss://' + MAY : (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host) +
    '/ws?k=' + encodeURIComponent(MA);
  ws = new WebSocket(u);
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => { trangThai('Đã kết nối', 'var(--xanh)'); idDaXong = idKhung; ketNoiLuc = phanHoiCuoi = Date.now();
    ws.send(JSON.stringify({ lenh: 'bat_dau' })); };   // cầu nối ESP32 ảo: bật dẫn đường (máy chủ thử bỏ qua)
  ws.onmessage = e => { if (typeof e.data === 'string') nhanJson(JSON.parse(e.data)); else nhanWav(e.data); };
  ws.onclose = e => {
    if (e.code === 1012) {   // máy chủ đã nhận kết nối mới hơn (tab / máy khác): dừng hẳn, không nối lại để khỏi đá nhau
      dung(); trangThai('Đã mở ở tab hoặc máy khác – trang này đã dừng', 'var(--do)'); return;
    }
    saiMa = e.code === 1008;
    trangThai(e.code === 1008 ? 'Sai mã – quét lại mã QR' : 'Mất kết nối, đang nối lại…', 'var(--do)');
    if (chay && cheDo === 'truc_tiep' && e.code !== 1008) setTimeout(() => chay && moKetNoi(), 2000);
  };
}

function guiKhung() {
  if (!chay || cheDo !== 'truc_tiep' || !ws || ws.readyState !== 1) return;
  if (Date.now() - guiLuc < 1000 / FPS - 8) return;   // không nhanh hơn FPS
  const v = $('xem');
  if (!v.videoWidth || v.videoHeight > v.videoWidth) return;   // cầm dọc: ảnh bị kéo méo -> khoảng cách sai, không gửi
  if (idKhung - idDaXong >= soKhungTrenDuong && Date.now() - guiLuc < CHO_TOI_DA_MS) return;
  if (ws.bufferedAmount > 0) return;   // khung trước còn nằm trong bộ đệm gửi: thêm khung chỉ làm hàng dài ra
  const cv = $('cv'), cx = cv.getContext('2d');
  if (KHUNG_43) {                      // cắt giữa phần 4:3 của khung 16:9 (góc nhìn ngang hẹp lại ~55°)
    const sw = v.videoHeight * 4 / 3;
    cx.drawImage(v, (v.videoWidth - sw) / 2, 0, sw, v.videoHeight, 0, 0, CW, CH);
  } else cx.drawImage(v, 0, 0, CW, CH);
  const t_chup = Date.now(), id = ++idKhung;
  cv.toBlob(async b => {
    if (!b || !ws || ws.readyState !== 1) return;
    const jpg = await b.arrayBuffer(), goi = new Uint8Array(12 + jpg.byteLength), dv = new DataView(goi.buffer);
    dv.setFloat64(0, t_chup, true); dv.setUint32(8, id, true); goi.set(new Uint8Array(jpg), 12);
    ws.send(goi.buffer);
  }, 'image/jpeg', 0.7);
  guiLuc = t_chup;
}

function canhGac() {   // chạy độc lập với máy tính: mất phản hồi -> tự báo động
  if (!chay || cheDo !== 'truc_tiep') return;
  const now = Date.now();
  // chưa gửi được khung nào kể từ lúc (nối lại): cho camera 3 s khởi động; đã gửi thì 1 s không phản hồi là báo
  const han = guiLuc > ketNoiLuc ? MAT_KET_NOI_MS : 3000;
  if (now - Math.max(phanHoiCuoi, ketNoiLuc) > han) {
    dangMat = true;
    trangThai(saiMa ? 'SAI MÃ – quét lại mã QR' : 'MẤT KẾT NỐI', 'var(--do)');
    if (now - baoDongLuc > 4000) { baoDongLuc = now; baoDong('mat_ket_noi'); }
  }
}

let dangBatDau = false;
async function giuSang() {
  try {
    const k = await navigator.wakeLock.request('screen');
    if (!chay && !dangBatDau) { k.release().catch(() => {}); return; }   // đã bấm Dừng trong lúc chờ khóa
    wakeLock = k; canh('');
  }
  catch (e) { canh('Không giữ được màn hình sáng: vào Cài đặt > Màn hình > Tự động khóa > Không bao giờ'); }
}

async function batDau() {
  cheDo = document.querySelector('input[name=cd]:checked').value;
  if (!MA && cheDo === 'truc_tiep') {   // mở link trần (vd. trang Pages): chưa biết máy tính nào -> đừng bật camera
    canh('Chưa nối máy tính: bật chương trình trên máy tính rồi quét mã QR.'); return;
  }
  dangBatDau = true;
  try { await batDauThat(); } finally { dangBatDau = false; }
}

async function batDauThat() {
  // Xin giữ màn hình sáng NGAY lúc chạm và KHÔNG chờ: WebKit chỉ cho xin trong vài giây sau lần chạm (hỏi quyền
  // camera lâu là bị từ chối), và lời xin có thể treo tới lần chạm sau (đo trên WebKit 2026-11-10: treo ~19 s) ->
  // chờ nó thì camera đã bật mà chưa nối máy tính, không lời dẫn, không báo động.
  giuSang();
  $('cam').src = amCam(); await $('cam').play().catch(() => {});
  ac = ac || new (window.AudioContext || window.webkitAudioContext)();
  await ac.resume();
  ac.onstatechange = () => { if (chay && ac.state !== 'running') batLaiTieng(); };
  await Promise.all(['mat_ket_noi', 'trang_bi_tat', 'da_ket_noi'].map(napAm));
  luong = await navigator.mediaDevices.getUserMedia({ audio: false,
    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } } });
  $('xem').srcObject = luong;
  chay = true; document.body.classList.add('chay');
  $('nut').textContent = $('nut').dataset.dung || 'Dừng'; $('nut').classList.add('dung');
  if (cheDo === 'chi_ghi') {
    $('khongcb').style.display = 'block';
    const loai = ['video/mp4', 'video/webm'].find(t => MediaRecorder.isTypeSupported(t)) || '';
    ghi = new MediaRecorder(luong, loai ? { mimeType: loai } : undefined);
    manh = []; ghi.ondataavailable = e => e.data.size && manh.push(e.data); ghi.start(1000);
    trangThai('Đang ghi (không cảnh báo)', 'var(--vang)');
  } else {
    phanHoiCuoi = Date.now(); moKetNoi();
  }
}

function dung() {
  chay = false; document.body.classList.remove('chay');
  $('nut').textContent = $('nut').dataset.batDau || 'Bắt đầu'; $('nut').classList.remove('dung'); $('khongcb').style.display = 'none';
  if (ws) { if (ws.readyState === 1) ws.send(JSON.stringify({ lenh: 'dung' })); ws.onclose = null; ws.close(); ws = null; }
  if (ghi && ghi.state !== 'inactive') {
    ghi.onstop = () => {
      const b = new Blob(manh, { type: ghi.mimeType }), a = document.createElement('a');
      a.href = URL.createObjectURL(b);
      a.download = 'kinh_' + new Date().toISOString().replace(/[:.]/g, '-') + (ghi.mimeType.includes('mp4') ? '.mp4' : '.webm');
      a.click();
    };
    ghi.stop();
  }
  if (luong) luong.getTracks().forEach(t => t.stop());
  if (wakeLock) wakeLock.release().catch(() => {});
  $('cam').pause();
  trangThai('Đã dừng', 'var(--xam)');
}

$('nut').onclick = () => (chay ? dung() : batDau().catch(e => { canh('Lỗi: ' + e.message); dung(); }));
document.addEventListener('visibilitychange', () => {
  if (!chay) return;
  if (document.hidden) baoDong('trang_bi_tat');   // Safari sắp dừng trang: báo ngay khi còn kịp
  else { giuSang(); batLaiTieng(); }
});

function batLaiTieng() {
  if (!chay || !ac) return;
  $('cam').play().catch(() => {});
  ac.resume().catch(() => {}).finally(() => { $('battieng').style.display = ac.state === 'running' ? 'none' : 'flex'; });
}
if (!MA) canh('Thiếu mã: hãy mở bằng mã QR trên máy tính');
$('cv').width = CW; $('cv').height = CH;
// xét thường xuyên hơn FPS để có chỗ trống là gửi ngay: chờ đủ một nhịp thì khung cách nhau 250–375 ms khi máy tính
// xử lý gần bằng nhịp gửi (đo 2026-11-06: máy tính đều 115 ms mà ảnh về cách nhau tới 376 ms)
setInterval(guiKhung, 1000 / FPS / 4);
setInterval(canhGac, 250);
