using BenchmarkLite
using Winston

import JuliaMM
import CPUMM
import CudaMM

Benchmarks = [
  CudaMM,
  JuliaMM,
  CPUMM
  
];

Types = [
  Int32, Float32, Int64, Float64
];

function construct(benchmarks, types)

  vcat(map(function(b)
    map(function(t)
      b.Benchmark{t}()
    end, types)
  end, benchmarks)...)

end

procs = construct(Benchmarks, Types)

rtable = run(procs, cfgs);
println(typeof(rtable))
show(rtable; unit=:msec);

function save_plot(b::BenchmarkTable, filename)

  sizes = b.cfgs

  plot()
  for i in 1:size(b.etime, 2)


    oplot(sizes, (b.etime[:,i] ./ b.nruns[:,i]));
  end

  savefig(filename)

end

save_plot(rtable, "result.png")
