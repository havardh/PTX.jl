#include <iostream>

#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"

namespace llvm {
  extern Pass* createEmitJuliaDeclarationsPass();
}

using namespace llvm;

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Wrong number of arguments" << std::endl;
    exit(1);
  }

  LLVMContext Context;
  SMDiagnostic Err;
  Module* module = ParseIRFile(argv[1], Err, Context);

  if (module == NULL) {
    std::cout << "Error parsing module: " << argv[1] << std::endl;
    exit(1);
  }

  PassManager PM;
  PM.add(createEmitJuliaDeclarationsPass());
  PM.run(*module);

  return 0;
}
