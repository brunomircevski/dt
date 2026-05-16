#include "tree_visualization.h"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {

std::string shellEscapeSingleQuoted(const std::string &value) {
  std::string escaped = "'";
  for (char character : value) {
    if (character == '\'') {
      escaped += "'\\''";
    } else {
      escaped += character;
    }
  }
  escaped += "'";
  return escaped;
}

} // namespace

void generateTreeSvg(const C45Tree &tree, const std::string &svgPath) {
  std::ostringstream printedTree;
  tree.print(printedTree);

  const std::string treeTextPath = "tree_visualization.txt";
  std::ofstream output(treeTextPath);
  if (!output) {
    throw std::runtime_error("Could not write visualization input file.");
  }

  output << printedTree.str();
  output.close();

  const std::string command =
      "python3 scripts/render_tree_svg.py " +
      shellEscapeSingleQuoted(treeTextPath) + " " +
      shellEscapeSingleQuoted(svgPath);

  const int status = std::system(command.c_str());
  if (status != 0) {
    throw std::runtime_error("Could not generate tree SVG.");
  }
}
