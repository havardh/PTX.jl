#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Constants.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

struct EmitJuliaDeclarationsPass : public ModulePass {

  static char ID;
  EmitJuliaDeclarationsPass() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {


    errs() << "module OpenCL\n";
    errs() << "function Void() end\n";
    
    for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
      if (!I->isIntrinsic()) {
        emitDeclaration(*I);
      }
    }
    errs() << "\n\nend\n";
    
    return false;
  }

  void emitDeclaration(Function &F) {

    
    FunctionType *FunctionType = F.getFunctionType();
    Type *ReturnType = F.getReturnType();

    if (ReturnType->isVoidTy()) {
      errs() << "@noinline "
             << F.getName()
             << "(";
      emitArguments(FunctionType);
      errs() << ") = Void()\n";
    } else {

      errs() << "@noinline "
             << F.getName()
             << "(";
      emitArguments(FunctionType);
      errs() << ") = zero(";
      emitType(ReturnType);
      errs() << ")\n";
    }
    
  }

  void emitArguments(FunctionType* FunctionType) {

    for (int i=0; i<FunctionType->getNumParams(); i++) {
      errs() << (i==0? "" : ", ")
             << "::";
      emitType(FunctionType->getParamType(i));
    }

  }

  void emitType(Type* Type) {
    LLVMContext& Context = Type->getContext();

    
    if (Type == Type::getInt1Ty(Context)) {
      errs() << "Bool";
    } else if (Type == Type::getInt8Ty(Context)) {
      errs() << "Int8";
    } else if (Type == Type::getInt16Ty(Context)) {
      errs() << "Int16";
    } else if (Type == Type::getInt32Ty(Context)) {
      errs() << "Int32";
    } else if (Type == Type::getInt64Ty(Context)) {
      errs() << "Int64";
    } else if (Type == Type::getFloatTy(Context)) {
      errs() << "Float32";
    } else if (Type == Type::getDoubleTy(Context)) {
      errs() << "Float64";
    } else if (Type->isPointerTy() || Type->isVectorTy() || Type->isStructTy()) {
      errs() << "Bool";
    } else {
      errs() << "Could not recognize type: " << *Type << "\n";
      exit(1);
    }
  }

};

char EmitJuliaDeclarationsPass::ID = 0;

static RegisterPass<EmitJuliaDeclarationsPass> X("Emit Julia Declarations Pass", "Emits definitions for Julia functions", false, false);

Pass* createEmitJuliaDeclarationsPass() {
  return new EmitJuliaDeclarationsPass();
}
  
}
