import { Suspense } from "react";
import { Card } from "@/components/ui/Card";
import { SessionPickerClient } from "@/components/race/SessionPickerClient";

export default function RacePage() {
  return (
    <div className="flex flex-col gap-6">
      <h2 className="text-xl font-bold tracking-tight">Race Dashboard</h2>
      <Card>
        <Suspense>
          <SessionPickerClient />
        </Suspense>
      </Card>
    </div>
  );
}
