#include <iostream>

#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Analysis/Verifier.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Bitcode/ReaderWriter.h"

namespace llvm {
  extern Pass* createLowerJuliaArrayPass();
  extern Pass* createFunctionInliningPass();
  extern Pass* createAnnotateKernelsPass();
}

using namespace llvm;

int main(int argc, char **argv) {
  
  if (argc != 2) {
    std::cout << "Wrong number of argumnts" << std::endl;
    exit(1);
  }

  LLVMContext Context;
  SMDiagnostic Err;
  Module *module = ParseIRFile(argv[1], Err, Context);

  if (module == NULL) {
    Err.print("", errs());
    exit(1);
  }
  
  if (verifyModule(*module, PrintMessageAction)) {
    std::cout << "Input module does not validate" << std::endl;
    exit(1);
  }

  module->setDataLayout(StringRef("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"));
  module->setTargetTriple(StringRef("nvptx64-nvidia-cuda"));
  
  llvm::PassManager modulePassManager;
  modulePassManager.add(createLowerJuliaArrayPass());
  modulePassManager.add(createFunctionInliningPass());
  modulePassManager.add(createAnnotateKernelsPass());
  modulePassManager.run(*module);

  if (verifyModule(*module, PrintMessageAction)) {
    std::cout << "Transformed module does not validate" << std::endl;
    exit(1);
  }

  module->dump();
    
  return 0;
}
