'use client';

import { HugeiconsIcon } from '@hugeicons/react';
import type { IconSvgElement } from '@hugeicons/react';
import {
  Camera01Icon,
  CameraOff01Icon,
  FlipVerticalIcon,
  Moon02Icon,
  Refresh01Icon,
  RulerIcon,
  SparklesIcon,
  Sun03Icon,
  SwitchCameraIcon,
  Video01Icon,
} from '@hugeicons/core-free-icons';

type AppIconProps = {
  icon: IconSvgElement;
  size?: number;
  strokeWidth?: number;
  className?: string;
};

export function AppIcon({
  icon,
  size = 16,
  strokeWidth = 1.8,
  className,
}: AppIconProps) {
  return (
    <HugeiconsIcon
      icon={icon}
      size={size}
      color="currentColor"
      strokeWidth={strokeWidth}
      className={className}
    />
  );
}

export const icons = {
  live: Video01Icon,
  photo: Camera01Icon,
  cameraOff: CameraOff01Icon,
  flip: FlipVerticalIcon,
  guides: RulerIcon,
  camera: SwitchCameraIcon,
  retake: Refresh01Icon,
  sparkles: SparklesIcon,
  sun: Sun03Icon,
  moon: Moon02Icon,
} as const;
