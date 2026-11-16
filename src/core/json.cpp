#include "core/json.h"

#include <charconv>
#include <cstdint>
#include <stdexcept>

namespace dt {

const Json *Json::find(std::string_view key) const {
  for (const auto &[name, value] : members) {
    if (name == key) {
      return &value;
    }
  }
  return nullptr;
}

namespace {

class Parser {
public:
  explicit Parser(std::string_view document) : text_(document) {}

  Json document() {
    Json value = parseValue(0);
    skipSpace();
    if (at_ != text_.size()) {
      fail("text after the JSON value");
    }
    return value;
  }

private:
  // Deeper nesting is an error rather than a stack overflow.
  static constexpr int kMaxDepth = 64;

  [[noreturn]] void fail(const std::string &what) const {
    throw std::runtime_error("Invalid JSON at byte " + std::to_string(at_) + ": " + what);
  }

  void skipSpace() {
    while (at_ < text_.size() &&
           (text_[at_] == ' ' || text_[at_] == '\t' || text_[at_] == '\n' || text_[at_] == '\r')) {
      ++at_;
    }
  }

  void expect(char c) {
    skipSpace();
    if (at_ >= text_.size() || text_[at_] != c) {
      fail(std::string("expected '") + c + "'");
    }
    ++at_;
  }

  // A ',' after optional white space (another member or item follows).
  bool consumeComma() {
    skipSpace();
    return consume(",");
  }

  bool consume(std::string_view word) {
    if (text_.substr(at_, word.size()) != word) {
      return false;
    }
    at_ += word.size();
    return true;
  }

  Json parseValue(int depth) {
    if (depth > kMaxDepth) {
      fail("nested too deeply");
    }
    skipSpace();
    if (at_ >= text_.size()) {
      fail("unexpected end");
    }
    Json value;
    const char c = text_[at_];
    if (c == '{') {
      ++at_;
      value.type = Json::Type::Object;
      skipSpace();
      if (at_ < text_.size() && text_[at_] == '}') {
        ++at_;
        return value;
      }
      do {
        skipSpace();
        std::string key = parseString();
        expect(':');
        value.members.emplace_back(std::move(key), parseValue(depth + 1));
      } while (consumeComma());
      expect('}');
    } else if (c == '[') {
      ++at_;
      value.type = Json::Type::Array;
      skipSpace();
      if (at_ < text_.size() && text_[at_] == ']') {
        ++at_;
        return value;
      }
      do {
        value.items.push_back(parseValue(depth + 1));
      } while (consumeComma());
      expect(']');
    } else if (c == '"') {
      value.type = Json::Type::String;
      value.text = parseString();
    } else if (consume("true") || consume("false")) {
      value.type = Json::Type::Bool;
      value.boolean = c == 't';
    } else if (consume("null")) {
      value.type = Json::Type::Null;
    } else {
      value.type = Json::Type::Number;
      const auto result =
          std::from_chars(text_.data() + at_, text_.data() + text_.size(), value.number);
      if (result.ec != std::errc{}) {
        fail("expected a value");
      }
      at_ = static_cast<std::size_t>(result.ptr - text_.data());
    }
    return value;
  }

  unsigned hex4() {
    if (at_ + 4 > text_.size()) {
      fail("short \\u escape");
    }
    unsigned code = 0;
    const auto result = std::from_chars(text_.data() + at_, text_.data() + at_ + 4, code, 16);
    if (result.ec != std::errc{} || result.ptr != text_.data() + at_ + 4) {
      fail("bad \\u escape");
    }
    at_ += 4;
    return code;
  }

  static void appendUtf8(std::string &out, std::uint32_t code) {
    if (code < 0x80) {
      out += static_cast<char>(code);
    } else if (code < 0x800) {
      out += static_cast<char>(0xC0 | (code >> 6));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else if (code < 0x10000) {
      out += static_cast<char>(0xE0 | (code >> 12));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (code >> 18));
      out += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (code & 0x3F));
    }
  }

  std::string parseString() {
    if (at_ >= text_.size() || text_[at_] != '"') {
      fail("expected a string");
    }
    ++at_;
    std::string out;
    while (true) {
      if (at_ >= text_.size()) {
        fail("unterminated string");
      }
      const char c = text_[at_++];
      if (c == '"') {
        return out;
      }
      if (c != '\\') {
        out += c;
        continue;
      }
      if (at_ >= text_.size()) {
        fail("unterminated string");
      }
      switch (const char escaped = text_[at_++]) {
      case '"': case '\\': case '/': out += escaped; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        std::uint32_t code = hex4();
        // Python's json.dump writes characters outside the BMP as a surrogate pair.
        if (code >= 0xD800 && code < 0xDC00 && consume("\\u")) {
          const std::uint32_t low = hex4();
          if (low < 0xDC00 || low >= 0xE000) {
            fail("bad surrogate pair");
          }
          code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        appendUtf8(out, code);
        break;
      }
      default:
        fail("bad escape");
      }
    }
  }

  std::string_view text_;
  std::size_t at_ = 0;
};

} // namespace

Json parseJson(std::string_view document) { return Parser(document).document(); }

} // namespace dt
