#pragma once

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dt {

// A parsed JSON value: just enough JSON for a binary dataset's meta.json (see
// loadDataset), so the build needs no JSON library.
struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool boolean = false;
  double number = 0.0;
  std::string text;                                 // String
  std::vector<Json> items;                          // Array
  std::vector<std::pair<std::string, Json>> members; // Object, in file order

  // The member `key` of an object, or null if there is none (or this is not an object).
  const Json *find(std::string_view key) const;
};

// Parses a whole JSON document (UTF-8; \u escapes become UTF-8). Throws
// std::runtime_error with the byte offset of the first error.
Json parseJson(std::string_view document);

} // namespace dt
