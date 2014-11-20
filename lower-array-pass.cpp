#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/IPO/InlinerPass.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Regex.h"
#include "llvm/Linker.h"
#include "llvm/IR/Constants.h"

#include "linker.h"

namespace llvm {
  
bool is_jl_array_type(Type* type) {
  if (type->isPointerTy()) {
    Type *elemType = type->getPointerElementType();
    if (elemType->isStructTy()) {
      if (elemType->getStructName() == StringRef("jl_value_t")) {
        return true;
      }
    }
  }
  return false;
}

struct LowerJuliaArrayPass : public ModulePass {

  static char ID;
  LowerJuliaArrayPass() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {
    
    std::vector<Function*> Fs = functions(M);
    
    linkLibrary(M);

    for (std::vector<Function*>::iterator I = Fs.begin(), E = Fs.end(); I != E; ++I) {

      Function *OldFunc = (*I);

      Function *NewFunc = copyFunctionWithLoweredJuliaArrayArguments(M, OldFunc);
      //OldFunc->replaceAllUsesWith(NewFunc);
      OldFunc->eraseFromParent();
      
      //if (!NewFunc->isDeclaration()) {
        generateFunctionMetadata(M, NewFunc);
      //}
    }
    
    return false;
  }

  std::vector<Function*> functions(Module &M) {
    std::vector<Function*> functions;

    for (Module::iterator F = M.begin(); F != M.end(); ++F) {

      if (!F->isDeclaration()) {
        functions.push_back(F);
      }
    }
    
    return functions;
  }

  void generateFunctionMetadata(Module &M, Function *F) {

    //generateOpenCLKernelMetadata(M, F);
    generateNVVMKernelMetadata(M, F);
    
    
  }

  void generateOpenCLKernelMetadata(Module &M, Function *F) {

    SmallVector <llvm::Value*, 5> kernelMDArgs;
    kernelMDArgs.push_back(F);

    MDNode *kernelMDNode = MDNode::get(M.getContext(), kernelMDArgs);
    
    NamedMDNode* OpenCLKernels = M.getOrInsertNamedMetadata("opencl.kernels");
    OpenCLKernels->addOperand(kernelMDNode);
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

  void linkLibrary(Module &M) {
    
    SMDiagnostic error;
    Module* LowLevelJuliaArray = ParseIRFile("/home/havard/projects/PTX.jl/lowered-julia-array.bc", error, M.getContext());
    Module* OpenCLPTXLibrary = ParseIRFile("/home/havard/projects/PTX.jl/nvptx64-nvidia-cuda.bc", error, M.getContext());

    link(&M, LowLevelJuliaArray);
    link(&M, OpenCLPTXLibrary);
    
  }

  Function* copyFunctionWithLoweredJuliaArrayArguments(Module &M, Function *OldFunc) {

    std::vector<Type*> ArgTypes = lowerJuliaArrayArguments(OldFunc);

    FunctionType *functionType = buildLoweredFunctionType(OldFunc, ArgTypes);

    Function* NewFunc = Function::Create(
      functionType,
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

    return NewFunc;
  }

  std::vector<Type*> lowerJuliaArrayArguments(Function *OldFunc) {

    std::vector<Type*> ArgTypes;
    for (Function::const_arg_iterator I = OldFunc->arg_begin(), E = OldFunc->arg_end(); I != E; ++I) {

      Type* argType = I->getType();

      if (is_jl_array_type(argType)) {
        // Should figure out actual type from meta?
        // This is hardcoded i64*
        ArgTypes.push_back(PointerType::get(IntegerType::get(OldFunc->getContext(), 64), 1));
      } else {
        ArgTypes.push_back(I->getType());
      }
      
    }

    return ArgTypes;
    
  }

  FunctionType* buildLoweredFunctionType(Function *F, std::vector<Type*> ArgTypes) {
    return FunctionType::get(
      Type::getVoidTy(F->getContext()),
      ArgTypes,
      F->getFunctionType()->isVarArg()
    );
  }
  
};

char LowerJuliaArrayPass::ID = 0;

static RegisterPass<LowerJuliaArrayPass> X("LowerJuliaArrayPass", "Lower Julia Array usage to c arrays", false, false);

Pass* createLowerJuliaArrayPass() {
  return new LowerJuliaArrayPass();
}

}
