import { PlotlyChart } from "@/components/charts/PlotlyChart";

const EMPTY_FIG = JSON.stringify({
  data: [],
  layout: {
    title: "Model Prediction Probabilities",
    xaxis: { title: "Probability", range: [0, 1] },
    yaxis: { title: "Driver" },
    height: 280,
  },
});

export function ProbabilityChart() {
  return (
    <div>
      <PlotlyChart figureJson={EMPTY_FIG} height={280} />
      <p className="mt-2 text-center text-xs italic text-white/40">
        Prediction probabilities not available in this build.
      </p>
    </div>
  );
}
