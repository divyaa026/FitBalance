import { useState } from 'react';
import { Button } from '@/components/ui/button';
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogHeader,
    DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select';
import { FitnessGoal } from '@/types/profile';

interface EditGoalDialogProps {
    goal?: FitnessGoal;
    onUpdate: (data: Partial<FitnessGoal>) => void;
    open: boolean;
    onOpenChange: (open: boolean) => void;
    isNewGoal?: boolean;
}

export function EditGoalDialog({ goal, onUpdate, open, onOpenChange, isNewGoal = false }: EditGoalDialogProps) {
    const [formData, setFormData] = useState({
        type: goal?.type || 'custom',
        name: goal?.name || '',
        target: goal?.target || 0,
        unit: goal?.unit || '',
        deadline: goal?.deadline || '',
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        onUpdate(formData);
        onOpenChange(false);
    };

    return (
        <Dialog open={open} onOpenChange={onOpenChange}>
            <DialogContent className="sm:max-w-[425px]">
                <DialogHeader>
                    <DialogTitle>{isNewGoal ? 'Create New Goal' : 'Edit Goal'}</DialogTitle>
                    <DialogDescription>
                        {isNewGoal ? 'Set a new fitness goal.' : 'Update your fitness goal.'}
                    </DialogDescription>
                </DialogHeader>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div className="space-y-2">
                        <Label htmlFor="type">Goal Type</Label>
                        <Select
                            value={formData.type}
                            onValueChange={(value) => setFormData({ ...formData, type: value as FitnessGoal['type'] })}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="Select type" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="nutrition">Nutrition</SelectItem>
                                <SelectItem value="workout">Workout</SelectItem>
                                <SelectItem value="form">Form</SelectItem>
                                <SelectItem value="custom">Custom</SelectItem>
                            </SelectContent>
                        </Select>
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="name">Goal Name</Label>
                        <Input
                            id="name"
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            placeholder="E.g., Daily Protein Intake"
                        />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <Label htmlFor="target">Target Value</Label>
                            <Input
                                id="target"
                                type="number"
                                value={formData.target}
                                onChange={(e) => setFormData({ ...formData, target: Number(e.target.value) })}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="unit">Unit</Label>
                            <Input
                                id="unit"
                                value={formData.unit}
                                onChange={(e) => setFormData({ ...formData, unit: e.target.value })}
                                placeholder="E.g., g, sessions"
                            />
                        </div>
                    </div>
                    <div className="space-y-2">
                        <Label htmlFor="deadline">Deadline (Optional)</Label>
                        <Input
                            id="deadline"
                            type="date"
                            value={formData.deadline}
                            onChange={(e) => setFormData({ ...formData, deadline: e.target.value })}
                        />
                    </div>
                    <div className="flex justify-end space-x-2">
                        <Button type="button" variant="secondary" onClick={() => onOpenChange(false)}>
                            Cancel
                        </Button>
                        <Button type="submit">{isNewGoal ? 'Create Goal' : 'Save Changes'}</Button>
                    </div>
                </form>
            </DialogContent>
        </Dialog>
    );
}