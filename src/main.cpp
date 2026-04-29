#include <iostream>
#include <sycl/sycl.hpp>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

/*#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>*/
#include <iostream>

#include <architecture/looping_retnet.hpp>
#include <long_term/graph.hpp>
#include <long_term/memory_module.hpp>
#include <long_term/spatial_map.hpp>
#include <tokenizer.hpp>
#include <training/looping_retnet_training.hpp>
/*
namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace json = boost::json;
using tcp = net::ip::tcp;

class WebSocketSession : public std::enable_shared_from_this<WebSocketSession> {
 public:
  explicit WebSocketSession(tcp::socket&& socket)
      : ws_(std::move(socket)) {}

  void run() {
    ws_.set_option(websocket::stream_base::timeout::suggested(beast::role_type::server));
    ws_.set_option(websocket::stream_base::decorator([](websocket::response_type& res) {
      res.set(beast::http::field::server, "llm_sycl_ws");
    }));

    ws_.async_accept(
      beast::bind_front_handler(&WebSocketSession::on_accept, shared_from_this())
    );
  }

 private:
  websocket::stream<beast::tcp_stream> ws_;
  beast::flat_buffer buffer_;
  std::deque<std::string> write_queue_;

  void on_accept(beast::error_code ec) {
    if (ec) {
      std::cerr << "Accept failed: " << ec.message() << "\n";
      return;
    }
    do_read();
  }

  void do_read() {
    ws_.async_read(
      buffer_,
      beast::bind_front_handler(&WebSocketSession::on_read, shared_from_this())
    );
  }

  void on_read(beast::error_code ec, std::size_t) {
    if (ec == websocket::error::closed) {
      return;
    }
    if (ec) {
      std::cerr << "Read failed: " << ec.message() << "\n";
      return;
    }

    std::string payload = beast::buffers_to_string(buffer_.data());
    buffer_.consume(buffer_.size());

    json::object response;
    response["ok"] = true;

    try {
      response["received"] = json::parse(payload);
    } catch (const std::exception& err) {
      response["ok"] = false;
      response["error"] = err.what();
      response["raw"] = payload;
    }

    enqueue_write(json::serialize(response));
  }

  void enqueue_write(std::string message) {
    bool writing = !write_queue_.empty();
    write_queue_.push_back(std::move(message));
    if (!writing) {
      do_write();
    }
  }

  void do_write() {
    ws_.text(true);
    ws_.async_write(
      net::buffer(write_queue_.front()),
      beast::bind_front_handler(&WebSocketSession::on_write, shared_from_this())
    );
  }

  void on_write(beast::error_code ec, std::size_t) {
    if (ec) {
      std::cerr << "Write failed: " << ec.message() << "\n";
      return;
    }

    write_queue_.pop_front();
    if (!write_queue_.empty()) {
      do_write();
      return;
    }

    do_read();
  }
};

class Listener : public std::enable_shared_from_this<Listener> {
 public:
  Listener(net::io_context& ioc, tcp::endpoint endpoint)
      : ioc_(ioc), acceptor_(net::make_strand(ioc)) {
    beast::error_code ec;

    acceptor_.open(endpoint.protocol(), ec);
    if (ec) {
      throw std::runtime_error("open: " + ec.message());
    }

    acceptor_.set_option(net::socket_base::reuse_address(true), ec);
    if (ec) {
      throw std::runtime_error("set_option: " + ec.message());
    }

    acceptor_.bind(endpoint, ec);
    if (ec) {
      throw std::runtime_error("bind: " + ec.message());
    }

    acceptor_.listen(net::socket_base::max_listen_connections, ec);
    if (ec) {
      throw std::runtime_error("listen: " + ec.message());
    }
  }

  void run() {
    do_accept();
  }

 private:
  net::io_context& ioc_;
  tcp::acceptor acceptor_;

  void do_accept() {
    acceptor_.async_accept(
      net::make_strand(ioc_),
      beast::bind_front_handler(&Listener::on_accept, shared_from_this())
    );
  }

  void on_accept(beast::error_code ec, tcp::socket socket) {
    if (!ec) {
      std::make_shared<WebSocketSession>(std::move(socket))->run();
    } else {
      std::cerr << "Incoming connection failed: " << ec.message() << "\n";
    }

    do_accept();
  }
};

void run_sycl_demo() {
  constexpr size_t n = 1024;

  std::vector<float> a(n, 1.5f);
  std::vector<float> b(n, 2.0f);
  std::vector<float> out(n, 0.0f);

  sycl::queue q{sycl::default_selector_v};
  std::cout << "SYCL device: "
            << q.get_device().get_info<sycl::info::device::name>()
            << '\n';

  sycl::buffer<float> a_buf(a.data(), sycl::range<1>(n));
  sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
  sycl::buffer<float> out_buf(out.data(), sycl::range<1>(n));

  q.submit([&](sycl::handler& h) {
    auto a_acc = a_buf.get_access<sycl::access::mode::read>(h);
    auto b_acc = b_buf.get_access<sycl::access::mode::read>(h);
    auto out_acc = out_buf.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      out_acc[i] = a_acc[i] + b_acc[i];
    });
  });

  q.wait_and_throw();

  std::cout << "Result sample: " << out[0] << " " << out[n / 2] << " " << out[n - 1] << '\n';
}

unsigned short get_port_from_env() {
  const char* env = std::getenv("SERVER_PORT");
  if (!env) {
    return 8080;
  }

  int value = std::atoi(env);
  if (value <= 0 || value > 65535) {
    return 8080;
  }

  return static_cast<unsigned short>(value);
}

net::ip::address get_host_from_env() {
  const char* env = std::getenv("SERVER_HOST");
  if (!env || std::string(env).empty()) {
    return net::ip::make_address("0.0.0.0");
  }

  beast::error_code ec;
  auto address = net::ip::make_address(env, ec);
  if (ec) {
    return net::ip::make_address("0.0.0.0");
  }

  return address;
}

void run_server() {
  try {
    const auto host = get_host_from_env();
    const unsigned short port = get_port_from_env();
    const auto endpoint = tcp::endpoint{host, port};

    net::io_context ioc;
    std::make_shared<Listener>(ioc, endpoint)->run();

    net::signal_set signals(ioc, SIGINT, SIGTERM);
    signals.async_wait([&](const beast::error_code&, int) {
      ioc.stop();
    });

    std::cout << "WebSocket JSON server listening on " << host.to_string() << ":" << port << '\n';
    ioc.run();
  } catch (const std::exception& err) {
    throw std::runtime_error(std::string("Server error: ") + err.what());
  }
}*/

namespace {

using llm::arch::LoopConfig;
using llm::arch::LoopingRetNet;
using llm::arch::Scalar;
using llm::training::looping::LoopingRetNetSGDTrainer;
using llm::training::looping::TrainConfig;
using llm::training::looping::TrainMode;
using llm::training::looping::TrainOptimizer;

uintmax_t safe_file_size(const std::string& path) {
  try {
    return std::filesystem::file_size(path);
  } catch (...) {
    return 0;
  }
}

class ProgressBar {
 public:
  ProgressBar(std::string label, uintmax_t total_bytes)
      : label_(std::move(label)), total_bytes_(total_bytes) {
    if (total_bytes_ > 0) {
      render(0);
    }
  }

  void update(uintmax_t processed_bytes) {
    if (total_bytes_ == 0 || finished_) {
      return;
    }
    if (processed_bytes > total_bytes_) {
      processed_bytes = total_bytes_;
    }
    const int pct = static_cast<int>((processed_bytes * 100u) / total_bytes_);
    if (pct == last_pct_) {
      return;
    }
    render(pct);
  }

  void finish() {
    if (total_bytes_ == 0 || finished_) {
      return;
    }
    render(100);
    std::cerr << "\n";
    finished_ = true;
  }

 private:
  void render(int pct) {
    static constexpr size_t kBarWidth = 30;
    const size_t filled = (static_cast<size_t>(pct) * kBarWidth) / 100u;
    std::cerr << "\r" << label_ << " [";
    for (size_t i = 0; i < kBarWidth; ++i) {
      std::cerr << (i < filled ? '=' : ' ');
    }
    std::cerr << "] " << pct << "%";
    std::cerr.flush();
    last_pct_ = pct;
  }

  std::string label_;
  uintmax_t total_bytes_ = 0;
  int last_pct_ = -1;
  bool finished_ = false;
};

struct ParsedArgs {
  std::vector<std::string> positionals;
  std::unordered_map<std::string, std::string> flags;
};

std::string to_lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

bool parse_bool(const std::string& value) {
  const std::string v = to_lower(value);
  if (v == "1" || v == "true" || v == "yes" || v == "on") {
    return true;
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    return false;
  }
  throw std::runtime_error("Invalid boolean value: " + value);
}

size_t parse_size(const std::string& value, const std::string& name) {
  try {
    const size_t idx = 0;
    size_t used = idx;
    const unsigned long long parsed = std::stoull(value, &used);
    if (used != value.size()) {
      throw std::runtime_error("");
    }
    return static_cast<size_t>(parsed);
  } catch (...) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

uint32_t parse_u32(const std::string& value, const std::string& name) {
  const size_t n = parse_size(value, name);
  if (n > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
    throw std::runtime_error("Out-of-range value for " + name + ": " + value);
  }
  return static_cast<uint32_t>(n);
}

float parse_float(const std::string& value, const std::string& name) {
  try {
    size_t used = 0;
    const float parsed = std::stof(value, &used);
    if (used != value.size()) {
      throw std::runtime_error("");
    }
    return parsed;
  } catch (...) {
    throw std::runtime_error("Invalid value for " + name + ": " + value);
  }
}

TrainMode parse_train_mode(const std::string& raw) {
  const std::string mode = to_lower(raw);
  if (mode == "finitedifference" || mode == "fd") {
    return TrainMode::FiniteDifference;
  }
  if (mode == "backpropheads" || mode == "heads") {
    return TrainMode::BackpropHeads;
  }
  if (mode == "backpropfull" || mode == "full") {
    return TrainMode::BackpropFull;
  }
  throw std::runtime_error("Unsupported train mode: " + raw);
}

TrainOptimizer parse_train_optimizer(const std::string& raw) {
  const std::string opt = to_lower(raw);
  if (opt == "adam") {
    return TrainOptimizer::Adam;
  }
  if (opt == "muon") {
    return TrainOptimizer::Muon;
  }
  if (opt == "sgd") {
    return TrainOptimizer::SGD;
  }
  throw std::runtime_error("Unsupported optimizer: " + raw);
}

ParsedArgs parse_args(int argc, char** argv, int start_index) {
  ParsedArgs out;
  for (int i = start_index; i < argc; ++i) {
    std::string token = argv[i];
    if (token.rfind("--", 0) == 0) {
      std::string key;
      std::string value;
      const size_t eq = token.find('=');
      if (eq != std::string::npos) {
        key = token.substr(2, eq - 2);
        value = token.substr(eq + 1);
      } else {
        key = token.substr(2);
        if (i + 1 < argc && std::string(argv[i + 1]).rfind("-", 0) != 0) {
          value = argv[++i];
        } else {
          value = "true";
        }
      }
      out.flags[key] = value;
      continue;
    }
    if (token.rfind("-", 0) == 0 && token.size() > 1 && token[1] != '-') {
      if (i + 1 >= argc) {
        throw std::runtime_error("Flag requires a value: " + token);
      }
      const std::string key = token.substr(1);
      out.flags[key] = argv[++i];
      continue;
    }
    out.positionals.push_back(token);
  }
  return out;
}

bool has_flag(const ParsedArgs& args, const std::string& long_name, const std::string& short_name = "") {
  return args.flags.find(long_name) != args.flags.end()
      || (!short_name.empty() && args.flags.find(short_name) != args.flags.end());
}

std::string get_flag(const ParsedArgs& args, const std::string& long_name, const std::string& short_name = "") {
  auto it = args.flags.find(long_name);
  if (it != args.flags.end()) {
    return it->second;
  }
  if (!short_name.empty()) {
    it = args.flags.find(short_name);
    if (it != args.flags.end()) {
      return it->second;
    }
  }
  return "";
}

std::vector<std::string> parse_csv_row(const std::string& line);

std::vector<std::string> load_dataset_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open dataset file: " + path);
  }

  ProgressBar progress("Loading dataset CSV", safe_file_size(path));
  uintmax_t fallback_bytes = 0;

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    fallback_bytes += static_cast<uintmax_t>(line.size() + 1);
    const std::streampos pos = in.tellg();
    if (pos >= 0) {
      progress.update(static_cast<uintmax_t>(pos));
    } else {
      progress.update(fallback_bytes);
    }

    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }

    const std::vector<std::string> cols = parse_csv_row(line);
    if (cols.size() < 3) {
      continue;
    }

    if (lines.empty()
        && to_lower(cols[0]) == "doc_id"
        && to_lower(cols[1]) == "sent_id"
        && to_lower(cols[2]) == "text") {
      continue;
    }

    if (!cols[2].empty()) {
      lines.push_back(cols[2]);
    }
  }

  progress.finish();

  if (lines.empty()) {
    throw std::runtime_error("Dataset CSV has no valid text rows (expected cols: doc_id,sent_id,text): " + path);
  }
  return lines;
}

std::vector<std::string> parse_csv_row(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  bool in_quotes = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char c = line[i];
    if (c == '"') {
      if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
        cur.push_back('"');
        ++i;
      } else {
        in_quotes = !in_quotes;
      }
      continue;
    }
    if (c == ',' && !in_quotes) {
      out.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  out.push_back(cur);
  return out;
}

struct SentenceCsvRow {
  std::string sentence;
  std::string k_text;
  std::string r_text;
  std::string v_text;
};

std::vector<SentenceCsvRow>
load_sentence_krv_csv_rows(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open sentence database CSV: " + path);
  }

  ProgressBar progress("Loading sentence CSV", safe_file_size(path));
  uintmax_t fallback_bytes = 0;

  std::vector<SentenceCsvRow> rows;
  std::string line;
  while (std::getline(in, line)) {
    fallback_bytes += static_cast<uintmax_t>(line.size() + 1);
    const std::streampos pos = in.tellg();
    if (pos >= 0) {
      progress.update(static_cast<uintmax_t>(pos));
    } else {
      progress.update(fallback_bytes);
    }

    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      continue;
    }
    const std::vector<std::string> cols = parse_csv_row(line);
    if (cols.size() < 4) {
      continue;
    }
    if (rows.empty() && to_lower(cols[0]) == "sentence") {
      continue;
    }
    SentenceCsvRow ex;
    ex.sentence = cols[0];
    ex.k_text = cols[1];
    ex.r_text = cols[2];
    ex.v_text = cols[3];
    rows.push_back(std::move(ex));
  }

  progress.finish();

  if (rows.empty()) {
    throw std::runtime_error("Sentence database CSV has no valid rows: " + path);
  }
  return rows;
}

float fp16_to_float(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  const uint32_t exp = (h >> 10) & 0x1Fu;
  const uint32_t frac = h & 0x03FFu;

  uint32_t bits = 0;
  if (exp == 0) {
    if (frac == 0) {
      bits = sign;
    } else {
      int e = -14;
      uint32_t f = frac;
      while ((f & 0x0400u) == 0) {
        f <<= 1;
        --e;
      }
      f &= 0x03FFu;
      bits = sign | (static_cast<uint32_t>(e + 127) << 23) | (f << 13);
    }
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (frac << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (frac << 13);
  }

  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t read_u16_le(std::istream& in) {
  uint8_t b[2] = {0, 0};
  in.read(reinterpret_cast<char*>(b), 2);
  return static_cast<uint16_t>(static_cast<uint16_t>(b[0])
      | (static_cast<uint16_t>(b[1]) << 8));
}

uint64_t read_u64_le(std::istream& in) {
  uint8_t b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  in.read(reinterpret_cast<char*>(b), 8);
  uint64_t v = 0;
  for (size_t i = 0; i < 8; ++i) {
    v |= (static_cast<uint64_t>(b[i]) << (8 * i));
  }
  return v;
}

struct KRVEmbeddingRecord {
  std::vector<float> k;
  std::vector<float> r;
  std::vector<float> v;
};

std::vector<KRVEmbeddingRecord> load_krv_embedding_database(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open KRV database: " + path);
  }

  ProgressBar progress("Loading KRV binary", safe_file_size(path));
  uintmax_t fallback_bytes = 0;

  constexpr uint64_t kMagic = 0x4B5256454D424544ULL; // KRVEMBED
  const uint64_t magic = read_u64_le(in);
  fallback_bytes += 8;
  if (!in || magic != kMagic) {
    throw std::runtime_error("Invalid KRV database magic in: " + path);
  }

  const uint16_t k_len = read_u16_le(in);
  const uint16_t r_len = read_u16_le(in);
  const uint16_t v_len = read_u16_le(in);
  (void)read_u16_le(in); // padding
  fallback_bytes += 8;
  progress.update(fallback_bytes);
  if (!in || k_len == 0 || r_len == 0 || v_len == 0) {
    throw std::runtime_error("Invalid KRV database header in: " + path);
  }

  std::vector<KRVEmbeddingRecord> rows;
  while (true) {
    KRVEmbeddingRecord rec;
    rec.k.resize(k_len);
    rec.r.resize(r_len);
    rec.v.resize(v_len);

    bool ok = true;
    for (size_t i = 0; i < k_len; ++i) {
      if (!in.good()) { ok = false; break; }
      rec.k[i] = fp16_to_float(read_u16_le(in));
      fallback_bytes += 2;
    }
    for (size_t i = 0; ok && i < r_len; ++i) {
      if (!in.good()) { ok = false; break; }
      rec.r[i] = fp16_to_float(read_u16_le(in));
      fallback_bytes += 2;
    }
    for (size_t i = 0; ok && i < v_len; ++i) {
      if (!in.good()) { ok = false; break; }
      rec.v[i] = fp16_to_float(read_u16_le(in));
      fallback_bytes += 2;
    }

    if (!ok) {
      break;
    }

    const std::streampos pos = in.tellg();
    if (pos >= 0) {
      progress.update(static_cast<uintmax_t>(pos));
    } else {
      progress.update(fallback_bytes);
    }

    rows.push_back(std::move(rec));
  }

  progress.finish();

  if (rows.empty()) {
    throw std::runtime_error("KRV database contains no embedding rows: " + path);
  }
  return rows;
}

std::vector<Scalar> sentence_embedding_hash(const std::string& text, size_t dim) {
  std::vector<float> tmp(dim, 0.0f);
  if (dim == 0 || text.empty()) {
    return std::vector<Scalar>(dim, static_cast<Scalar>(0.0f));
  }

  for (size_t i = 0; i < text.size(); ++i) {
    const uint32_t c = static_cast<uint8_t>(text[i]);
    const uint32_t h1 = 2166136261u ^ (c + static_cast<uint32_t>(i * 131u));
    const uint32_t h2 = (h1 * 16777619u) ^ (c * 1315423911u);
    const size_t idx1 = static_cast<size_t>(h1 % static_cast<uint32_t>(dim));
    const size_t idx2 = static_cast<size_t>(h2 % static_cast<uint32_t>(dim));
    tmp[idx1] += 1.0f;
    tmp[idx2] -= 0.5f;
  }

  float n2 = 0.0f;
  for (float v : tmp) {
    n2 += v * v;
  }
  const float n = std::sqrt(std::max(1e-12f, n2));
  std::vector<Scalar> out(dim, static_cast<Scalar>(0.0f));
  for (size_t i = 0; i < dim; ++i) {
    out[i] = static_cast<Scalar>(tmp[i] / n);
  }
  return out;
}

std::vector<Scalar> fit_embedding_dim(const std::vector<float>& in, size_t dim) {
  std::vector<Scalar> out(dim, static_cast<Scalar>(0.0f));
  const size_t n = std::min(dim, in.size());
  for (size_t i = 0; i < n; ++i) {
    out[i] = static_cast<Scalar>(in[i]);
  }
  return out;
}

// Remove spaces that appear immediately before punctuation characters.
std::string strip_spaces_before_punct(const std::string& s) {
  static const std::string punct = ".!?,;:'\")]}";
  std::string out;
  out.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == ' ' && i + 1 < s.size() && punct.find(s[i + 1]) != std::string::npos) {
      continue;
    }
    out += s[i];
  }
  return out;
}

std::string get_memory_module_flag(const ParsedArgs& args) {
  std::string path = get_flag(args, "memory_module", "m");
  if (!path.empty()) {
    return path;
  }
  return get_flag(args, "memory-module", "m");
}

std::vector<std::vector<size_t>> tokenize_texts(
    const std::vector<std::string>& texts,
    const Tokenizer& tokenizer) {
  std::vector<std::vector<size_t>> out;
  out.reserve(texts.size());
  for (const auto& text : texts) {
    const auto toks = tokenizer.tokenize(text, TokenizationMode::Training);
    std::vector<size_t> ids;
    ids.reserve(toks.size());
    for (const auto& t : toks) {
      ids.push_back(t.id);
    }
    out.push_back(std::move(ids));
  }
  return out;
}

void write_memory_module_from_dataset(
    const LoopingRetNet& model,
    const std::vector<llm::training::looping::SequenceExample>& dataset,
    const llm::memory::MemoryConfig& mem_cfg,
    uint32_t seed,
    const std::string& memory_module_path) {
  if (memory_module_path.empty()) {
    return;
  }

  Graph graph;
  SpatialMap spatial_map;
  llm::memory::NodeCompressor compressor(model.config().v_dim, mem_cfg.semvec_dim, seed);
  llm::memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), mem_cfg);
  llm::memory::MultiHopQuery query(graph, spatial_map, mem_cfg);

  llm::arch::KVState recurrent_state;
  llm::arch::AttentionMemory chrono_kv_cache;
  llm::arch::AttentionMemory krv_cache;

  constexpr bool kForceOutput = false;
  constexpr bool kEnableQuery = true;
  constexpr size_t kForcedLoops = 1;
  constexpr bool kUseParallelRetention = true;
  constexpr bool kEnableMemoryWrite = true;
  constexpr bool kForceQueryFirst = false;

  LoopingRetNet model_copy = model;
  for (const auto& ex : dataset) {
    const size_t steps = std::min(ex.input.size(), ex.target.size());
    for (size_t i = 0; i < steps; ++i) {
      (void)model_copy.step_with_trace(
          ex.input[i],
          recurrent_state,
          chrono_kv_cache,
          krv_cache,
          bridge,
          query,
          kForceOutput,
          kEnableQuery,
          kForcedLoops,
          kUseParallelRetention,
          kEnableMemoryWrite,
          kForceQueryFirst);
    }
  }

  graph.save_to_file(memory_module_path);
}

void print_usage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program << " server\n"
      << "  " << program << " train -d DATASET -o OUTPUT [options]\n"
      << "  " << program << " infer -i MODEL [options]\n\n"
      << "Train required flags:\n"
      << "  -d, --dataset PATH            CSV dataset with columns: doc_id,sent_id,text\n"
      << "  -o, --output PATH             Output model binary path\n"
      << "  -tok, --tokenizer PATH        Tokenizer vocab CSV (token,score)\n"
      << "  -m, --memory_module PATH      Save trained memory module graph path\n\n"
      << "Train common optional flags:\n"
      << "  -i, --input PATH              Existing model binary to continue training\n"
      << "  --mode MODE                   FiniteDifference|BackpropHeads|BackpropFull\n"
      << "  --optimizer OPT              Adam|Muon|SGD\n"
      << "  --epochs N                    Training epochs\n"
      << "  --learning-rate F             Learning rate\n"
      << "  --batch-size N                Batch size\n"
      << "  --host-threads N              Host CPU worker threads (0 = auto)\n"
      << "  --min-parallel-batch N        Min batch examples before threading\n"
      << "  --weight-decay F              Weight decay\n"
      << "  --sgd-momentum F              SGD momentum\n"
      << "  --muon-momentum F             Muon momentum\n"
      << "  --muon-eps F                  Muon epsilon\n"
      << "  --samples-per-epoch N         Random samples per epoch (0 = use all)\n"
      << "  --force-query-prob F          Probability of forcing first inner step to QUERY_MEMORY\n"
      << "  --load-gate-supervision-weight F Weight for memory-load gate supervision\n"
      << "  --enable-action-renorm BOOL   Continuously renormalize action levels during training\n"
      << "  --target-queries-per-output F Target avg queries per emitted char (default: 0.05)\n"
      << "  --target-creates-per-output F Target avg memory creates per emitted char (default: 0.02)\n"
      << "  --action-renorm-gain F        Controller gain for adaptive action renormalization\n"
      << "  --action-renorm-max-bias F    Clamp for per-action renormalization bias\n"
      << "  -s, --s-database PATH         CSV sentence database (cols: sentence,K,R,V)\n"
      << "  -krv, --krv-database PATH     Binary KRV embedding DB aligned by sentence index\n"
      << "  --sentence-krv-loss-weight F  Weight for sentence->KRV alignment objective\n"
      << "  --model-dim N --qk-dim N --v-dim N --rel-dim N\n"
      << "  --hidden-layers N --max-steps N --decay F --model-seed N --train-seed N\n\n"
      << "Infer flags:\n"
      << "  -i, --input PATH              Model binary\n"
      << "  -tok, --tokenizer PATH        Tokenizer vocab CSV (token,score)\n"
      << "  -m, --memory_module PATH      Load memory module graph path\n"
      << "  -p, --prompt TEXT             Prompt text (default: empty)\n"
      << "  -t, --tokens N                Number of generated tokens (default: 128)\n"
      << "  --enable-query BOOL           Allow QUERY_MEMORY action during inference\n"
      << "  --force-output BOOL           Force OUTPUT action each step\n"
      << "  --use-parallel-retention BOOL Use parallel retention path\n"
      << "\n"
      << "Example:\n"
      << "  " << program << " train -i model.bin -o trained.bin -d dataset.txt --epochs 4096 --mode BackpropFull\n";
}

int run_train_command(const ParsedArgs& args) {
  const std::string dataset_path = get_flag(args, "dataset", "d");
  if (dataset_path.empty()) {
    throw std::runtime_error("train requires -d/--dataset");
  }
  const std::string output_path = get_flag(args, "output", "o");
  if (output_path.empty()) {
    throw std::runtime_error("train requires -o/--output");
  }

  const std::string input_path = get_flag(args, "input", "i");
  const std::string tokenizer_path = get_flag(args, "tokenizer", "tok");
  if (tokenizer_path.empty()) {
    throw std::runtime_error("train requires -tok/--tokenizer");
  }
  const Tokenizer tokenizer(tokenizer_path);

  const std::string memory_module_path = get_memory_module_flag(args);
  const std::string s_database_path = get_flag(args, "s-database", "s");
  const std::string krv_database_path = get_flag(args, "krv-database", "krv");
  if (!krv_database_path.empty() && s_database_path.empty()) {
    throw std::runtime_error("-krv/--krv-database requires -s/--s-database");
  }

  LoopConfig model_cfg;
  model_cfg.char_vocab = tokenizer.vocab_size();
  if (has_flag(args, "char-vocab")) {
    model_cfg.char_vocab = parse_size(get_flag(args, "char-vocab"), "char-vocab");
  }
  if (model_cfg.char_vocab != tokenizer.vocab_size()) {
    throw std::runtime_error("--char-vocab must match tokenizer vocab size");
  }
  if (has_flag(args, "model-dim")) {
    model_cfg.model_dim = parse_size(get_flag(args, "model-dim"), "model-dim");
  }
  if (has_flag(args, "qk-dim")) {
    model_cfg.qk_dim = parse_size(get_flag(args, "qk-dim"), "qk-dim");
  }
  if (has_flag(args, "v-dim")) {
    model_cfg.v_dim = parse_size(get_flag(args, "v-dim"), "v-dim");
  }
  if (has_flag(args, "rel-dim")) {
    model_cfg.rel_dim = parse_size(get_flag(args, "rel-dim"), "rel-dim");
  }
  if (has_flag(args, "hidden-layers")) {
    model_cfg.hidden_layers = parse_size(get_flag(args, "hidden-layers"), "hidden-layers");
  }
  if (has_flag(args, "decay")) {
    model_cfg.decay = parse_float(get_flag(args, "decay"), "decay");
  }
  if (has_flag(args, "max-steps")) {
    model_cfg.max_steps = parse_size(get_flag(args, "max-steps"), "max-steps");
  }
  const uint32_t model_seed = has_flag(args, "model-seed")
      ? parse_u32(get_flag(args, "model-seed"), "model-seed")
      : 42u;

  LoopingRetNet model = input_path.empty()
      ? LoopingRetNet(model_cfg, model_seed)
      : LoopingRetNet::load_from_file(input_path);
  if (model.config().char_vocab != tokenizer.vocab_size()) {
    throw std::runtime_error("Model vocabulary size does not match tokenizer vocab size");
  }

  TrainConfig train_cfg;
  train_cfg.memory_cfg.semvec_dim = model.config().v_dim;

  if (has_flag(args, "mode")) {
    train_cfg.mode = parse_train_mode(get_flag(args, "mode"));
  }
  if (has_flag(args, "optimizer")) {
    train_cfg.optimizer = parse_train_optimizer(get_flag(args, "optimizer"));
  }
  if (has_flag(args, "epochs")) {
    train_cfg.epochs = parse_size(get_flag(args, "epochs"), "epochs");
  }
  if (has_flag(args, "learning-rate")) {
    train_cfg.learning_rate = parse_float(get_flag(args, "learning-rate"), "learning-rate");
  }
  if (has_flag(args, "min-learning-rate")) {
    train_cfg.min_learning_rate = parse_float(get_flag(args, "min-learning-rate"), "min-learning-rate");
  }
  if (has_flag(args, "min-learning-rate-ratio")) {
    train_cfg.min_learning_rate_ratio = parse_float(get_flag(args, "min-learning-rate-ratio"), "min-learning-rate-ratio");
  }
  if (has_flag(args, "warmup-epochs")) {
    train_cfg.warmup_epochs = parse_size(get_flag(args, "warmup-epochs"), "warmup-epochs");
  }
  if (has_flag(args, "warmup-start-ratio")) {
    train_cfg.warmup_start_ratio = parse_float(get_flag(args, "warmup-start-ratio"), "warmup-start-ratio");
  }
  if (has_flag(args, "loss-ema-beta")) {
    train_cfg.loss_ema_beta = parse_float(get_flag(args, "loss-ema-beta"), "loss-ema-beta");
  }
  if (has_flag(args, "weight-decay")) {
    train_cfg.weight_decay = parse_float(get_flag(args, "weight-decay"), "weight-decay");
  }
  if (has_flag(args, "sgd-momentum")) {
    train_cfg.sgd_momentum = parse_float(get_flag(args, "sgd-momentum"), "sgd-momentum");
  }
  if (has_flag(args, "muon-momentum")) {
    train_cfg.muon_momentum = parse_float(get_flag(args, "muon-momentum"), "muon-momentum");
  }
  if (has_flag(args, "muon-eps")) {
    train_cfg.muon_eps = parse_float(get_flag(args, "muon-eps"), "muon-eps");
  }
  if (has_flag(args, "fd-eps")) {
    train_cfg.fd_eps = parse_float(get_flag(args, "fd-eps"), "fd-eps");
  }
  if (has_flag(args, "grad-coordinate-samples")) {
    train_cfg.grad_coordinate_samples = parse_size(get_flag(args, "grad-coordinate-samples"), "grad-coordinate-samples");
  }
  if (has_flag(args, "min-grad-coordinate-samples")) {
    train_cfg.min_grad_coordinate_samples = parse_size(get_flag(args, "min-grad-coordinate-samples"), "min-grad-coordinate-samples");
  }
  if (has_flag(args, "batch-size")) {
    train_cfg.batch_size = parse_size(get_flag(args, "batch-size"), "batch-size");
  }
  if (has_flag(args, "host-threads")) {
    train_cfg.host_threads = parse_size(get_flag(args, "host-threads"), "host-threads");
  }
  if (has_flag(args, "min-parallel-batch")) {
    train_cfg.min_parallel_batch_examples = parse_size(get_flag(args, "min-parallel-batch"), "min-parallel-batch");
  }
  if (has_flag(args, "backprop-force-single-step")) {
    train_cfg.backprop_force_single_step = parse_bool(get_flag(args, "backprop-force-single-step"));
  }
  if (has_flag(args, "backprop-include-loop-supervision")) {
    train_cfg.backprop_include_loop_supervision = parse_bool(get_flag(args, "backprop-include-loop-supervision"));
  }
  if (has_flag(args, "backprop-fd-check-samples")) {
    train_cfg.backprop_fd_check_samples = parse_size(get_flag(args, "backprop-fd-check-samples"), "backprop-fd-check-samples");
  }
  if (has_flag(args, "backprop-fd-check-eps")) {
    train_cfg.backprop_fd_check_eps = parse_float(get_flag(args, "backprop-fd-check-eps"), "backprop-fd-check-eps");
  }
  if (has_flag(args, "force-output")) {
    train_cfg.force_output = parse_bool(get_flag(args, "force-output"));
  }
  if (has_flag(args, "enable-query")) {
    train_cfg.enable_query = parse_bool(get_flag(args, "enable-query"));
  }
  if (has_flag(args, "use-parallel-retention")) {
    train_cfg.use_parallel_retention = parse_bool(get_flag(args, "use-parallel-retention"));
  }
  if (has_flag(args, "forced-loop-min")) {
    train_cfg.forced_loop_min = parse_size(get_flag(args, "forced-loop-min"), "forced-loop-min");
  }
  if (has_flag(args, "forced-loop-max")) {
    train_cfg.forced_loop_max = parse_size(get_flag(args, "forced-loop-max"), "forced-loop-max");
  }
  if (has_flag(args, "train-seed")) {
    train_cfg.seed = parse_u32(get_flag(args, "train-seed"), "train-seed");
  }
  if (has_flag(args, "samples-per-epoch")) {
    train_cfg.samples_per_epoch = parse_size(get_flag(args, "samples-per-epoch"), "samples-per-epoch");
  }
  if (has_flag(args, "force-query-prob")) {
    train_cfg.force_query_prob = parse_float(get_flag(args, "force-query-prob"), "force-query-prob");
  }
  if (has_flag(args, "load-gate-supervision-weight")) {
    train_cfg.load_gate_supervision_weight = parse_float(get_flag(args, "load-gate-supervision-weight"), "load-gate-supervision-weight");
  }
  if (has_flag(args, "enable-action-renorm")) {
    train_cfg.enable_action_renorm = parse_bool(get_flag(args, "enable-action-renorm"));
  }
  if (has_flag(args, "target-queries-per-output")) {
    train_cfg.target_queries_per_output = parse_float(get_flag(args, "target-queries-per-output"), "target-queries-per-output");
  }
  if (has_flag(args, "target-creates-per-output")) {
    train_cfg.target_memory_creates_per_output = parse_float(get_flag(args, "target-creates-per-output"), "target-creates-per-output");
  }
  if (has_flag(args, "action-renorm-gain")) {
    train_cfg.action_renorm_gain = parse_float(get_flag(args, "action-renorm-gain"), "action-renorm-gain");
  }
  if (has_flag(args, "action-renorm-max-bias")) {
    train_cfg.action_renorm_max_bias = parse_float(get_flag(args, "action-renorm-max-bias"), "action-renorm-max-bias");
  }
  if (has_flag(args, "sentence-krv-loss-weight")) {
    train_cfg.sentence_krv_loss_weight = parse_float(get_flag(args, "sentence-krv-loss-weight"), "sentence-krv-loss-weight");
  }

  if (has_flag(args, "memory-semvec-dim")) {
    train_cfg.memory_cfg.semvec_dim = parse_size(get_flag(args, "memory-semvec-dim"), "memory-semvec-dim");
  }
  if (has_flag(args, "memory-max-hop-depth")) {
    train_cfg.memory_cfg.max_hop_depth = parse_size(get_flag(args, "memory-max-hop-depth"), "memory-max-hop-depth");
  }
  if (has_flag(args, "memory-max-hop-breadth")) {
    train_cfg.memory_cfg.max_hop_breadth = parse_size(get_flag(args, "memory-max-hop-breadth"), "memory-max-hop-breadth");
  }
  if (has_flag(args, "memory-max-write-entries")) {
    train_cfg.memory_cfg.max_write_entries = parse_size(get_flag(args, "memory-max-write-entries"), "memory-max-write-entries");
  }

  if (has_flag(args, "focal-gamma")) {
    train_cfg.focal_gamma = parse_float(get_flag(args, "focal-gamma"), "focal-gamma");
  }

  if (!s_database_path.empty() && !krv_database_path.empty()) {
    const std::vector<SentenceCsvRow> sentence_rows = load_sentence_krv_csv_rows(s_database_path);
    std::vector<KRVEmbeddingRecord> krv_rows;

    krv_rows = load_krv_embedding_database(krv_database_path);
    if (krv_rows.size() != sentence_rows.size()) {
      throw std::runtime_error(
          "KRV database row count does not match sentence database row count");
    }

    train_cfg.sentence_krv_examples.clear();
    train_cfg.sentence_krv_examples.reserve(sentence_rows.size());
    for (size_t i = 0; i < sentence_rows.size(); ++i) {
      llm::training::looping::SentenceKRVExample ex;
      ex.sentence = sentence_rows[i].sentence;
      {
        const auto toks = tokenizer.tokenize(ex.sentence, TokenizationMode::Training);
        ex.token_ids.reserve(toks.size());
        for (const auto& t : toks) {
          ex.token_ids.push_back(t.id);
        }
      }

      ex.key_embedding = fit_embedding_dim(krv_rows[i].k, model.config().qk_dim);
      ex.relation_embedding = fit_embedding_dim(krv_rows[i].r, model.config().rel_dim);
      ex.value_embedding = fit_embedding_dim(krv_rows[i].v, model.config().v_dim);

      train_cfg.sentence_krv_examples.push_back(std::move(ex));
    }
    std::cout << "Loaded sentence->KRV alignment examples: "
              << train_cfg.sentence_krv_examples.size() << "\n";
  }

    const std::vector<std::string> dataset_texts = load_dataset_lines(dataset_path);
    std::cout << "Loaded dataset rows: " << dataset_texts.size() << "\n";

    const auto tokenized_groups = tokenize_texts(dataset_texts, tokenizer);

  const auto dataset = llm::training::looping::make_shift_dataset(tokenized_groups);
  if (dataset.empty()) {
    throw std::runtime_error("Dataset has no sequences with at least 2 tokens");
  }

  std::cout << "Training examples: " << dataset.size() << "\n";
  std::cout << "Model params: " << model.parameter_count() << "\n";

  LoopingRetNetSGDTrainer trainer(train_cfg);
  const auto history = trainer.train(model, dataset);
  if (history.empty()) {
    throw std::runtime_error("Training produced empty history");
  }

  model.save_to_file(output_path);
  if (!memory_module_path.empty()) {
    llm::memory::MemoryConfig export_mem_cfg = train_cfg.memory_cfg;
    export_mem_cfg.semvec_dim = model.config().v_dim;
    write_memory_module_from_dataset(
        model,
        dataset,
        export_mem_cfg,
        train_cfg.seed,
        memory_module_path);
    std::cout << "Saved memory module to: " << memory_module_path << "\n";
  }
  std::cout << "Final loss: " << std::setprecision(7) << history.back().avg_loss << "\n";
  std::cout << "Saved model to: " << output_path << "\n";
  return 0;
}

int run_infer_command(const ParsedArgs& args) {
  const std::string model_path = get_flag(args, "input", "i");
  if (model_path.empty()) {
    throw std::runtime_error("infer requires -i/--input");
  }
  const std::string memory_module_path = get_memory_module_flag(args);
  const std::string tokenizer_path = get_flag(args, "tokenizer", "tok");
  if (tokenizer_path.empty()) {
    throw std::runtime_error("infer requires -tok/--tokenizer");
  }
  const Tokenizer tokenizer(tokenizer_path);
  const std::string prompt = get_flag(args, "prompt", "p");
  const size_t tokens = has_flag(args, "tokens", "t")
      ? parse_size(get_flag(args, "tokens", "t"), "tokens")
      : 128;

  const bool enable_query = has_flag(args, "enable-query")
      ? parse_bool(get_flag(args, "enable-query"))
      : true;
  const bool force_output = has_flag(args, "force-output")
      ? parse_bool(get_flag(args, "force-output"))
      : false;
  const bool use_parallel_retention = has_flag(args, "use-parallel-retention")
      ? parse_bool(get_flag(args, "use-parallel-retention"))
      : false;

  LoopingRetNet model = LoopingRetNet::load_from_file(model_path);
  if (model.config().char_vocab != tokenizer.vocab_size()) {
    throw std::runtime_error("Model vocabulary size does not match tokenizer vocab size");
  }

  llm::memory::MemoryConfig mem_cfg;
  mem_cfg.semvec_dim = model.config().v_dim;
  Graph graph;
  SpatialMap spatial_map;
  llm::memory::NodeCompressor compressor(model.config().v_dim, mem_cfg.semvec_dim);
  llm::memory::GraphMemoryBridge bridge(graph, spatial_map, std::move(compressor), mem_cfg);
  llm::memory::MultiHopQuery query(graph, spatial_map, mem_cfg);

  if (!memory_module_path.empty()) {
    graph.load_from_file(memory_module_path, spatial_map);
    std::cout << "Loaded memory module from: " << memory_module_path << "\n";
  }

  llm::arch::KVState recurrent_state;
  llm::arch::AttentionMemory chrono_kv_cache;
  llm::arch::AttentionMemory krv_cache;

  std::vector<size_t> prompt_ids;
  {
    const auto prompt_tokens = tokenizer.tokenize(prompt);
    prompt_ids.reserve(prompt_tokens.size());
    for (const auto& t : prompt_tokens) {
      prompt_ids.push_back(t.id);
    }
  }
  size_t current = prompt_ids.empty() ? 0 : prompt_ids.back();
  // BackpropFull trains with a single recurrent step and no KV-cache accumulation
  // (state = ret_norm + v_current).  To match that distribution at inference time we
  // must also use exactly one inner step (forced_loops=1) and suppress KV-cache
  // writes (enable_memory_write=false).  Without this the attention output is a
  // weighted blend of ALL past values which the model has never seen during training,
  // producing out-of-distribution states that consistently decode to the most-frequent
  // character (space).
  constexpr size_t kInferForcedLoops    = 1;     // one step = matches BackpropFull
  constexpr bool   kInferWriteMemory    = false; // no KV-cache accumulation
  for (size_t token_id : prompt_ids) {
    const auto step = model.step_with_trace(
        token_id,
        recurrent_state,
      chrono_kv_cache,
      krv_cache,
        bridge,
        query,
        force_output,
        enable_query,
        kInferForcedLoops,
        use_parallel_retention,
        kInferWriteMemory);
    (void)step;
    current = token_id;
  }

  std::vector<size_t> generated_ids;
  generated_ids.reserve(tokens);
  for (size_t i = 0; i < tokens; ++i) {
    const auto step = model.step_with_trace(
        current,
        recurrent_state,
      chrono_kv_cache,
      krv_cache,
        bridge,
        query,
        force_output,
        enable_query,
        kInferForcedLoops,
        use_parallel_retention,
        kInferWriteMemory);
    current = step.output_token;
    generated_ids.push_back(current);
  }

  const std::string generated = tokenizer.decode(generated_ids);

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Completion: " << generated << "\n";
  return 0;
}

} // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  const std::string cmd = argv[1];

  try {
    if (cmd == "server") {
      //run_server();
      return 0;
    }
    if (cmd == "train") {
      const ParsedArgs args = parse_args(argc, argv, 2);
      return run_train_command(args);
    }
    if (cmd == "infer") {
      const ParsedArgs args = parse_args(argc, argv, 2);
      return run_infer_command(args);
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
      print_usage(argv[0]);
      return 0;
    }

    std::cerr << "Unknown command: " << cmd << "\n\n";
    print_usage(argv[0]);
    return 1;
  } catch (const std::exception& err) {
    std::cerr << err.what() << "\n";
    return 1;
  }
} 
