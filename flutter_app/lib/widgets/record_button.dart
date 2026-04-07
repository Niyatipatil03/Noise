import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';

class RecordButton extends StatelessWidget {
  final bool isRecording;
  final bool enabled;
  final VoidCallback onTap;

  const RecordButton({
    super.key,
    required this.isRecording,
    required this.enabled,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: enabled ? onTap : null,
      child: Stack(
        alignment: Alignment.center,
        children: [
          // Outer pulse ring (only when recording)
          if (isRecording)
            Container(
              width: 120,
              height: 120,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFFC62828).withOpacity(0.2),
              ),
            ).animate(onPlay: (c) => c.repeat())
              .scale(begin: const Offset(1, 1), end: const Offset(1.4, 1.4), duration: 900.ms)
              .fadeOut(duration: 900.ms),

          // Middle ring
          if (isRecording)
            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: const Color(0xFFC62828).withOpacity(0.3),
              ),
            ).animate(onPlay: (c) => c.repeat())
              .scale(begin: const Offset(1, 1), end: const Offset(1.25, 1.25), duration: 900.ms, delay: 150.ms)
              .fadeOut(duration: 900.ms, delay: 150.ms),

          // Main button
          AnimatedContainer(
            duration: const Duration(milliseconds: 300),
            width: 84,
            height: 84,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: isRecording
                    ? [const Color(0xFFE53935), const Color(0xFFB71C1C)]
                    : enabled
                        ? [const Color(0xFF1E88E5), const Color(0xFF0D47A1)]
                        : [Colors.grey.shade700, Colors.grey.shade900],
              ),
              boxShadow: [
                BoxShadow(
                  color: (isRecording ? Colors.red : Colors.blue).withOpacity(0.5),
                  blurRadius: 20,
                  spreadRadius: 2,
                ),
              ],
            ),
            child: Icon(
              isRecording ? Icons.stop_rounded : Icons.mic_rounded,
              color: Colors.white,
              size: 40,
            ),
          ),
        ],
      ),
    );
  }
}
