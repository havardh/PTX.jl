using BenchmarkLite
using Winston

import JuliaMM
import CPUMM

procs = Proc[
  JuliaMM.Benchmark{Int32}(),
  JuliaMM.Benchmark{Float32}(),
  JuliaMM.Benchmark{Int64}(),
  JuliaMM.Benchmark{Float64}(),
  CPUMM.Benchmark{Int32}(),
  CPUMM.Benchmark{Float32}(),
  CPUMM.Benchmark{Int64}(),
  CPUMM.Benchmark{Float64}()
];

cfgs = 2 .^ [1:8]

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
