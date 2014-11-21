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


  
void replaceAllCallsWith(Function* OldFunc, Function* NewFunc) {
  
  for (Value::use_iterator I = OldFunc->use_begin(), E = OldFunc->use_end(); I != E; ++I) {

    if (CallInst* call = dyn_cast<CallInst>(*I)) {
    
      std::vector<Value*> args;
      for(int i=0; i<call->getNumArgOperands(); i++) {
        args.push_back(call->getArgOperand(i));
      }
      ArrayRef<Value*> Args(args);
  
      CallInst *newCall = CallInst::Create(NewFunc, Args);
      if (newCall->getType() != call->getType()) {
        if (call->use_begin() != call->use_end()) {
          errs() << "Cannot handle usage of non matching return types for " << *call->getType() << " and " << *newCall->getType() << "\n";
        }

        newCall->insertBefore(call);
        call->replaceAllUsesWith(newCall);
        call->eraseFromParent();
    
      } else {
        ReplaceInstWithInst(call, newCall);
      }
    }
  }
}

Type* extractType(LLVMContext& context, StringRef type) {
  APInt Val;
  errs() << type.substr(1,2) << "\n";
  errs() << type.substr(1,2).getAsInteger(10, Val) << "\n";
  if (!type.substr(1,2).getAsInteger(10, Val)) {

    uint64_t bits = Val.getLimitedValue();
    switch (type[0]) {
    case 'i': {
      return PointerType::get(IntegerType::get(context, bits), 1);
    }
    case 'f': {
      if (bits == 32) {
        return PointerType::get(Type::getFloatTy(context), 1);
      } else {
        return PointerType::get(Type::getDoubleTy(context), 1);
      }
    }
    default:
      errs() << "Not supported\n";
      exit(1);
    }
    
  }

  return NULL;
}
  
struct LowerJuliaArrayPass : public ModulePass {

  static char ID;
  LowerJuliaArrayPass() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {
    
    std::vector<Function*> Fs = functions(M);

    linkLibrary(M);
    
    for (std::vector<Function*>::iterator I = Fs.begin(), E = Fs.end(); I != E; ++I) {

      Function *OldFunc = (*I);
      StringRef name = OldFunc->getName();
      Function *NewFunc = copyFunctionWithLoweredJuliaArrayArguments(M, OldFunc);
      replaceAllCallsWith(OldFunc, NewFunc);

      OldFunc->eraseFromParent();
      NewFunc->setName(name);
      
      if (NewFunc->hasFnAttribute("kernel")) {
        generateFunctionMetadata(M, NewFunc);
      }
    }
    
    return false;
  }

  std::vector<Function*> functions(Module &M) {
    std::vector<Function*> functions;

    for (Module::iterator F = M.begin(); F != M.end(); ++F) {

      if (!F->isDeclaration()) {
        F->addFnAttr("kernel");
      }
      functions.push_back(F);
    }
    
    return functions;
  }

  void generateFunctionMetadata(Module &M, Function *F) {

    generateNVVMKernelMetadata(M, F);
    
    
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

    Module* M = OldFunc->getParent();
    LLVMContext &context = M->getContext();
    NamedMDNode* JuliaArgs = M->getOrInsertNamedMetadata("julia.args");
    MDNode *node = JuliaArgs->getOperand(0);
    
    int operand = 0;
    std::vector<Type*> ArgTypes;
    for (Function::const_arg_iterator I = OldFunc->arg_begin(), E = OldFunc->arg_end(); I != E; ++I) {

      Type* argType = I->getType();

      if (is_jl_array_type(argType)) {
        // Should figure out actual type from meta?
        // This is hardcoded i64*

        Value *value = node->getOperand(operand);
        if (MDString* mdstring = dyn_cast<MDString>(value)) {

          if (Type* type = extractType(context, mdstring->getString())) {
            ArgTypes.push_back(type);            
          } else {
            errs() << "Could not extract type: ";
            mdstring->print(errs());
            errs() << "\n";
            exit(1);
          }
        } else {
          errs() << "Could not extract type: ";
          value->print(errs());
          errs() << "\n";
          exit(1);
        }
        
        
      } else {
        ArgTypes.push_back(I->getType());
      }
      operand++;
    }

    return ArgTypes;
    
  }

  FunctionType* buildLoweredFunctionType(Function *F, std::vector<Type*> ArgTypes) {
    return FunctionType::get(
      F->getReturnType(),
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
