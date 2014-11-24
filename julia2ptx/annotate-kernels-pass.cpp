#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Constants.h"

namespace llvm {

struct AnnotateKernelsPass : public ModulePass {

  static char ID;
  AnnotateKernelsPass() : ModulePass(ID) {}
  
  virtual bool runOnModule(Module& M) {

    for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {

      if (F->hasFnAttribute("kernel")) {
        generateNVVMKernelMetadata(M, F);
      }
    }

    return false;
  }

  void generateNVVMKernelMetadata(Module &M, Function *F) {

    LLVMContext &Ctx = M.getContext();
    
    SmallVector <llvm::Value*, 5> kernelMDArgs;
    kernelMDArgs.push_back(F);
    kernelMDArgs.push_back(MDString::get(Ctx, "kernel"));
    kernelMDArgs.push_back(ConstantInt::get(llvm::Type::getInt32Ty(Ctx), 1));

    MDNode *kernelMDNode = MDNode::get(Ctx, kernelMDArgs);

    NamedMDNode* NvvmAnnotations = M.getOrInsertNamedMetadata("nvvm.annotations");
    NvvmAnnotations->addOperand(kernelMDNode);

  }
  
};

char AnnotateKernelsPass::ID = 0;

static RegisterPass<AnnotateKernelsPass> X("Annotate Kernels Pass", "Annotates all functions with attribute kernel to run on the GPU", false, false);

Pass* createAnnotateKernelsPass() {
  return new AnnotateKernelsPass();
}
  
}
