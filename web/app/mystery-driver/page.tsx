import { Card } from "@/components/ui/Card";
import { ModelAccuracyPanel } from "@/components/mystery/ModelAccuracyPanel";
import { MysteryClient } from "./MysteryClient";

export default function MysteryDriverPage() {
  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">
        Can the model identify the driver?
      </h2>

      <MysteryClient />

      <hr className="border-white/10" />

      <Card>
        <ModelAccuracyPanel />
      </Card>
    </div>
  );
}
