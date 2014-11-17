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
}

using namespace llvm;


void cloneFunctionIntoModule(Module &M, Function *OldFunc)
{
  Function* NewFunc = Function::Create(
    OldFunc->getFunctionType(),
    OldFunc->getLinkage(),
    OldFunc->getName(),
    &M
  );

  ValueToValueMapTy VMap;
  Function::arg_iterator DestI = NewFunc->arg_begin();
  for (Function::const_arg_iterator I = OldFunc->arg_begin(), E = OldFunc->arg_end(); I != E; ++I) {
    VMap[I] = DestI++;
  }


  SmallVector<ReturnInst*, 5> Returns;
    
  CloneFunctionInto(NewFunc, OldFunc, VMap, false, Returns, "", 0);
  
}


int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "Wrong number of argumnts" << std::endl;
    exit(1);
  }

  LLVMContext Context;

  SMDiagnostic Err;
  
  Module *input = ParseIRFile(argv[1], Err, Context);
  if (verifyModule(*input, PrintMessageAction)) {
    std::cout << "Something went wrong!" << std::endl;
    exit(1);
  }
  
  Module module("OpenCL", input->getContext());
  module.setDataLayout(StringRef("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"));
  module.setTargetTriple(StringRef("nvptx64-nvidia-cuda"));
  for (Module::iterator I = input->begin(), E = input->end(); I != E; ++I) {
    if (!I->isDeclaration()) {
      cloneFunctionIntoModule(module, I);
    }
  }
  
  llvm::PassManager modulePassManager;
  modulePassManager.add(createLowerJuliaArrayPass());
  modulePassManager.add(createFunctionInliningPass());
  modulePassManager.run(module);

  module.getFunction("getindex")->eraseFromParent();
  module.getFunction("setindex")->eraseFromParent();

  module.dump();
    
  return 0;
}
