import { ResponsiveHeatMap } from '@nivo/heatmap';

interface TorqueHeatmapProps {
    data: { [key: string]: any };
    jointName: string;
}

export function TorqueHeatmap({ data, jointName }: TorqueHeatmapProps) {
    const heatmapKey = `${jointName}_torque`;
    const rawData = data?.[heatmapKey];

    if (!rawData || !Array.isArray(rawData)) {
        return (
            <div className="flex items-center justify-center h-64 text-muted-foreground">
                No heatmap data available for {jointName}
            </div>
        );
    }

    // transform to nivo heatmap expected structure
    const transformedData = rawData.map((row: number[], rowIndex: number) => ({
        id: `Row ${rowIndex}`,
        data: row.map((value: number, colIndex: number) => ({ x: `T${colIndex}`, y: value }))
    }));

    return (
        <div className="h-64">
            <ResponsiveHeatMap
                data={transformedData as any}
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
                colors={{ type: 'diverging', scheme: 'red_yellow_blue', divergeAt: 0.5 }}
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

export default TorqueHeatmap;
