#include <iostream>
#include <sycl/sycl.hpp>
#include <csignal>
#include <cstdlib>
#include <deque>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/json.hpp>

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

int main() {
  try {
    run_sycl_demo();

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
    std::cerr << "SYCL error: " << err.what() << "\n";
    return 1;
  }

  return 0;
}
