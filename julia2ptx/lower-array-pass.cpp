#include "llvm/ADT/StringSwitch.h"
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

bool is_jl_array_type(Type*);
void replaceAllCallsWith(Value*, Value*);
Type* extractType(LLVMContext&, StringRef);
StringRef mangle(LLVMContext&,StringRef, FunctionType*);
  
struct LowerJuliaArrayPass : public ModulePass {

  static char ID;
  LowerJuliaArrayPass() : ModulePass(ID) {}

  virtual bool runOnModule(Module &M) {
    
    std::vector<Function*> Fs = functions(M);
    
    
    for (std::vector<Function*>::iterator I = Fs.begin(), E = Fs.end(); I != E; ++I) {

      Function *F = (*I);
      
      if (F->isDeclaration()) {
        replaceWithLoweredDeclaration(M, F);
      } else {
        replaceWithLoweredImplementation(M, F);
      }
    }

    linkLibrary(M);
    
    return false;
  }

  void replaceWithLoweredDeclaration(Module& M, Function* F) {

    LLVMContext& context = M.getContext();
    
    StringRef name = F->getName();
    if (!(name.equals("getindex") || name.equals("setindex"))) {
      return; 
    }
    
    std::vector<Type*> ArgTypes = lowerJuliaArrayArgumentsDecl(F);

    FunctionType *functionType = buildLoweredFunctionType(F, ArgTypes);

    Function* NewFunc = Function::Create(
      functionType,
      F->getLinkage(),
      F->getName(),
      &M
    );
    replaceAllCallsWith(F, NewFunc);
    F->eraseFromParent();
    NewFunc->setName(mangle(context, name, functionType));
    
  }

  void replaceWithLoweredImplementation(Module& M, Function* F) {

    std::vector<Type*> ArgTypes = lowerJuliaArrayArguments(F);
    Function* NewFunc = copyFunctionWithArguments(M, F, ArgTypes);

    StringRef name = F->getName();
    replaceAllCallsWith(F, NewFunc);
    F->eraseFromParent();
    NewFunc->setName(name);
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

  void linkLibrary(Module &M) {
    
    SMDiagnostic error;
    Module* LowLevelJuliaArray = ParseIRFile("/home/havard/projects/PTX.jl/julia2ptx/lowered-julia-array.bc", error, M.getContext());
    Module* OpenCLPTXLibrary = ParseIRFile("/home/havard/projects/PTX.jl/lib/nvptx64-nvidia-cuda.bc", error, M.getContext());

    link(&M, LowLevelJuliaArray);
    link(&M, OpenCLPTXLibrary);
  }

  Function* copyFunctionWithArguments(Module &M, Function *OldFunc, std::vector<Type*> ArgTypes) {

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
        
        // Gets the type from custom metadata
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

  std::vector<Type*> lowerJuliaArrayArgumentsDecl(Function* F) {
    std::vector<Type*> ArgTypes;
    Type* ReturnType = F->getReturnType();
    
    for (Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I) {

      Type* argType = I->getType();
      if (is_jl_array_type(argType)) {
        ArgTypes.push_back(PointerType::get(ReturnType, 1));
      } else {
        ArgTypes.push_back(I->getType());
      }
      
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


  
void replaceAllCallsWith(Value* OldFunc, Value* NewFunc) {
  
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
    } else {
      (*I)->print(errs()); errs() << "\n";
      exit(1);
    }
  }
}


Type* extractType(LLVMContext& context, StringRef type) {

  return PointerType::get(llvm::StringSwitch<llvm::Type*>(type)
    .Case("i32*", Type::getInt32Ty(context))
    .Case("i64*", Type::getInt64Ty(context))
    .Case("float*", Type::getFloatTy(context))
    .Case("double*", Type::getDoubleTy(context))
    .Default(NULL), 1);

}

StringRef mangle(LLVMContext& context, StringRef name, FunctionType* FT) {

  if (name.equals("getindex") || name.equals("setindex")) {

    char* types = new char[FT->getNumParams()+1];
    types[FT->getNumParams()] = '\0';
    for (int i=0; i<FT->getNumParams(); i++) {

      Type* type = FT->getParamType(i);

      if (type->isPointerTy()) {
        type = FT->getContainedType(0);
      }
      
      if (type == Type::getInt32Ty(context)) {
        types[i] = 'i';
      } else if (type == Type::getInt64Ty(context)) {
        types[i] = 'l';
      } else if (type == Type::getFloatTy(context)) {
        types[i] = 'f';
      } else if (type == Type::getDoubleTy(context)) {
        types[i] = 'd';
      } else {
        errs() << "Unknown type: "; FT->getParamType(i)->print(errs()); errs() << "\n";
        exit(1);
      }
      
    }

    return Twine("_Z8").concat(name).concat("PU3AS1").concat(types).str();
    
  } else {
    return name;
  }
}
  
}
