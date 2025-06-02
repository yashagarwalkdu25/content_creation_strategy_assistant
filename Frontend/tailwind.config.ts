
import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				// Cyberpunk color palette
				cyber: {
					black: '#000000',
					dark: '#0a0a0a',
					gray: '#1a1a1a',
					'gray-light': '#2a2a2a',
					neon: '#00ffff',
					'neon-blue': '#0099ff',
					'neon-purple': '#9d4edd',
					'neon-pink': '#ff006e',
					'neon-green': '#39ff14',
				},
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				}
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'glow-pulse': {
					'0%, 100%': {
						boxShadow: '0 0 20px rgba(0, 255, 255, 0.5)'
					},
					'50%': {
						boxShadow: '0 0 40px rgba(0, 255, 255, 0.8)'
					}
				},
				'slide-in-left': {
					from: {
						transform: 'translateX(-100%)',
						opacity: '0'
					},
					to: {
						transform: 'translateX(0)',
						opacity: '1'
					}
				},
				'fade-in-up': {
					from: {
						transform: 'translateY(20px)',
						opacity: '0'
					},
					to: {
						transform: 'translateY(0)',
						opacity: '1'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'glow-pulse': 'glow-pulse 2s ease-in-out infinite',
				'slide-in-left': 'slide-in-left 0.5s ease-out',
				'fade-in-up': 'fade-in-up 0.6s ease-out'
			},
			backdropBlur: {
				xs: '2px',
			},
			boxShadow: {
				'cyber': '0 0 20px rgba(0, 255, 255, 0.3)',
				'cyber-lg': '0 0 40px rgba(0, 255, 255, 0.4)',
				'glass': '0 8px 32px rgba(0, 0, 0, 0.37)',
			}
		}
	},
	plugins: [
		require("tailwindcss-animate"),
		function({ addUtilities }: any) {
			const newUtilities = {
				'.glass': {
					background: 'rgba(255, 255, 255, 0.05)',
					backdropFilter: 'blur(10px)',
					border: '1px solid rgba(255, 255, 255, 0.1)',
				},
				'.glass-dark': {
					background: 'rgba(0, 0, 0, 0.3)',
					backdropFilter: 'blur(10px)',
					border: '1px solid rgba(255, 255, 255, 0.1)',
				},
				'.neon-border': {
					border: '1px solid #00ffff',
					boxShadow: '0 0 10px rgba(0, 255, 255, 0.5)',
				},
				'.neon-glow': {
					boxShadow: '0 0 20px rgba(0, 255, 255, 0.6)',
				}
			}
			addUtilities(newUtilities)
		}
	],
} satisfies Config;
