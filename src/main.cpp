#include <iostream>
#include <sycl/sycl.hpp>
#include <csignal>
#include <cstdlib>
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
using llm::training::looping::LoopingRetNetSGDTrainer;
using llm::training::looping::TrainConfig;
using llm::training::looping::TrainMode;

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
    if (token.rfind("-", 0) == 0 && token.size() == 2) {
      if (i + 1 >= argc) {
        throw std::runtime_error("Flag requires a value: " + token);
      }
      const std::string key(1, token[1]);
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

std::vector<std::string> load_dataset_lines(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open dataset file: " + path);
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (!line.empty()) {
      lines.push_back(line);
    }
  }

  if (lines.empty()) {
    throw std::runtime_error("Dataset file has no non-empty lines: " + path);
  }
  return lines;
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
  llm::arch::AttentionMemory kv_cache;

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
          kv_cache,
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

// Group lines into multi-sentence samples, randomly choosing a count in
// [min_sentences, max_sentences] per group, then join with a space.
std::vector<std::string> group_sentences(
    const std::vector<std::string>& lines,
    size_t min_sentences,
    size_t max_sentences,
    uint32_t seed)
{
  if (min_sentences < 1) min_sentences = 1;
  if (max_sentences < min_sentences) max_sentences = min_sentences;

  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> cnt_dist(min_sentences, max_sentences);

  std::vector<std::string> groups;
  size_t i = 0;
  while (i < lines.size()) {
    const size_t n = std::min(cnt_dist(rng), lines.size() - i);
    std::string group;
    for (size_t j = 0; j < n; ++j) {
      if (j > 0) group += ' ';
      group += strip_spaces_before_punct(lines[i + j]);
    }
    groups.push_back(std::move(group));
    i += n;
  }
  return groups;
}

void print_usage(const char* program) {
  std::cout
      << "Usage:\n"
      << "  " << program << " server\n"
      << "  " << program << " train -d DATASET -o OUTPUT [options]\n"
      << "  " << program << " infer -i MODEL [options]\n\n"
      << "Train required flags:\n"
      << "  -d, --dataset PATH            Plain-text dataset (one sentence per line)\n"
      << "  -o, --output PATH             Output model binary path\n"
      << "  -m, --memory_module PATH      Save trained memory module graph path\n\n"
      << "Train common optional flags:\n"
      << "  -i, --input PATH              Existing model binary to continue training\n"
      << "  --mode MODE                   FiniteDifference|BackpropHeads|BackpropFull\n"
      << "  --epochs N                    Training epochs\n"
      << "  --learning-rate F             Learning rate\n"
      << "  --batch-size N                Batch size\n"
      << "  --host-threads N              Host CPU worker threads (0 = auto)\n"
      << "  --min-parallel-batch N        Min batch examples before threading\n"
      << "  --weight-decay F              Weight decay\n"
      << "  --sgd-momentum F              SGD momentum\n"
      << "  --samples-per-epoch N         Random samples per epoch (0 = use all)\n"
      << "  --force-query-prob F          Probability of forcing first inner step to QUERY_MEMORY\n"
      << "  --load-gate-supervision-weight F Weight for memory-load gate supervision\n"
      << "  --group-min-sentences N       Min sentences per training sample (default: 3)\n"
      << "  --group-max-sentences N       Max sentences per training sample (default: 5)\n"
      << "  --group-seed N                RNG seed for sentence grouping (default: 0)\n"
      << "  --model-dim N --qk-dim N --v-dim N --rel-dim N\n"
      << "  --hidden-layers N --max-steps N --decay F --model-seed N --train-seed N\n\n"
      << "Infer flags:\n"
      << "  -i, --input PATH              Model binary\n"
      << "  -m, --memory_module PATH      Load memory module graph path\n"
      << "  -p, --prompt TEXT             Prompt text (default: empty)\n"
      << "  -t, --tokens N                Number of generated chars (default: 128)\n"
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
  const std::string memory_module_path = get_memory_module_flag(args);

  LoopConfig model_cfg;
  if (has_flag(args, "char-vocab")) {
    model_cfg.char_vocab = parse_size(get_flag(args, "char-vocab"), "char-vocab");
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

  TrainConfig train_cfg;
  train_cfg.memory_cfg.semvec_dim = model.config().v_dim;

  if (has_flag(args, "mode")) {
    train_cfg.mode = parse_train_mode(get_flag(args, "mode"));
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

  const std::vector<std::string> raw_lines = load_dataset_lines(dataset_path);
  std::cout << "Loaded raw lines: " << raw_lines.size() << "\n";

  const size_t group_min = has_flag(args, "group-min-sentences")
      ? parse_size(get_flag(args, "group-min-sentences"), "group-min-sentences")
      : 3;
  const size_t group_max = has_flag(args, "group-max-sentences")
      ? parse_size(get_flag(args, "group-max-sentences"), "group-max-sentences")
      : 5;
  const uint32_t group_seed = has_flag(args, "group-seed")
      ? parse_u32(get_flag(args, "group-seed"), "group-seed")
      : 0u;

  const std::vector<std::string> grouped_lines =
      group_sentences(raw_lines, group_min, group_max, group_seed);
  std::cout << "After grouping: " << grouped_lines.size() << " samples\n";

  const auto dataset = llm::training::looping::make_shift_dataset(grouped_lines);
  if (dataset.empty()) {
    throw std::runtime_error("Dataset has no sequences with at least 2 characters");
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
  llm::arch::AttentionMemory kv_cache;

  char current = prompt.empty() ? ' ' : prompt.back();
  // BackpropFull trains with a single recurrent step and no KV-cache accumulation
  // (state = ret_norm + v_current).  To match that distribution at inference time we
  // must also use exactly one inner step (forced_loops=1) and suppress KV-cache
  // writes (enable_memory_write=false).  Without this the attention output is a
  // weighted blend of ALL past values which the model has never seen during training,
  // producing out-of-distribution states that consistently decode to the most-frequent
  // character (space).
  constexpr size_t kInferForcedLoops    = 1;     // one step = matches BackpropFull
  constexpr bool   kInferWriteMemory    = false; // no KV-cache accumulation
  for (char c : prompt) {
    const auto step = model.step_with_trace(
        c,
        recurrent_state,
        kv_cache,
        bridge,
        query,
        force_output,
        enable_query,
        kInferForcedLoops,
        use_parallel_retention,
        kInferWriteMemory);
    (void)step;
    current = c;
  }

  std::string generated;
  generated.reserve(tokens);
  for (size_t i = 0; i < tokens; ++i) {
    const auto step = model.step_with_trace(
        current,
        recurrent_state,
        kv_cache,
        bridge,
        query,
        force_output,
        enable_query,
        kInferForcedLoops,
        use_parallel_retention,
        kInferWriteMemory);
    current = step.output_char;
    generated.push_back(current);
  }

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
