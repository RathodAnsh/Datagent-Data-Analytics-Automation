import { Loader2 } from 'lucide-react';

interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'success' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  disabled?: boolean;
  icon?: React.ReactNode;
  fullWidth?: boolean;
}

export default function Button({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  icon,
  fullWidth = false,
}: ButtonProps) {
  const baseClasses = 'font-medium rounded-lg transition-all flex items-center justify-center gap-2';

  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 shadow-md hover:shadow-lg disabled:bg-blue-300',
    secondary: 'bg-gray-600 text-white hover:bg-gray-700 shadow-md hover:shadow-lg disabled:bg-gray-300',
    success: 'bg-green-600 text-white hover:bg-green-700 shadow-md hover:shadow-lg disabled:bg-green-300',
    outline: 'bg-white text-gray-700 border-2 border-gray-300 hover:border-gray-400 hover:bg-gray-50 disabled:bg-gray-100',
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-5 py-2.5 text-base',
    lg: 'px-6 py-3 text-lg',
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${
        fullWidth ? 'w-full' : ''
      } disabled:cursor-not-allowed`}
    >
      {loading && <Loader2 className="w-5 h-5 animate-spin" />}
      {!loading && icon && icon}
      {children}
    </button>
  );
}
