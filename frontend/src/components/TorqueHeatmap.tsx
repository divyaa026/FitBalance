import { ResponsiveHeatMap } from '@nivo/heatmap';

interface TorqueHeatmapProps {
  data: { [key: string]: number[][] };
  jointName: string;
}

export function TorqueHeatmap({ data, jointName }: TorqueHeatmapProps) {
  // Transform data for nivo heatmap
  const heatmapKey = `${jointName}_torque`;
  const rawData = data[heatmapKey];
  
  if (!rawData || !Array.isArray(rawData)) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No heatmap data available for {jointName}
      </div>
    );
  }

  const transformedData = rawData.map((row, rowIndex) => ({
    id: `Row ${rowIndex}`,
    data: Array.isArray(row) 
      ? row.map((value, colIndex) => ({
          x: `Col ${colIndex}`,
          y: value
        }))
      : []
  }));

  return (
    <div className="h-64">
      <ResponsiveHeatMap
        data={transformedData}
        margin={{ top: 10, right: 10, bottom: 40, left: 40 }}
        valueFormat=">-.2f"
        axisTop={null}
        axisRight={null}
        axisBottom={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Time',
          legendPosition: 'middle',
          legendOffset: 32
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          tickRotation: 0,
          legend: 'Torque',
          legendPosition: 'middle',
          legendOffset: -30
        }}
        colors={{
          type: 'diverging',
          scheme: 'red_yellow_blue',
          divergeAt: 0.5
        }}
        emptyColor="#555555"
        legends={[
          {
            anchor: 'bottom',
            translateX: 0,
            translateY: 30,
            length: 400,
            thickness: 8,
            direction: 'row',
            tickPosition: 'after',
            tickSize: 3,
            tickSpacing: 4,
            tickOverlap: false,
            tickFormat: '>-.2s',
            title: 'Torque (Nm) →',
            titleAlign: 'start',
            titleOffset: 4
          }
        ]}
      />
    </div>
  );
}
