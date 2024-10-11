package cn.edu.hitsz.compiler.lexer;

import cn.edu.hitsz.compiler.NotImplementedException;
import cn.edu.hitsz.compiler.symtab.SymbolTable;
import cn.edu.hitsz.compiler.utils.FileUtils;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.StreamSupport;

/**
 * TODO: 实验一: 实现词法分析
 * <br>
 * 你可能需要参考的框架代码如下:
 *
 * @see Token 词法单元的实现
 * @see TokenKind 词法单元类型的实现
 */
public class LexicalAnalyzer {
    private final SymbolTable symbolTable;
    private String sourceCode;
    private final List<Token> tokens = new ArrayList<>();
    public LexicalAnalyzer(SymbolTable symbolTable) {
        this.symbolTable = symbolTable;
    }


    /**
     * 从给予的路径中读取并加载文件内容
     *
     * @param path 路径
     */
    public void loadFile(String path) {
        // TODO: 词法分析前的缓冲区实现
        // 可自由实现各类缓冲区
        // 或直接采用完整读入方法
        sourceCode = FileUtils.readFile(path);
    }

    /**
     * 执行词法分析, 准备好用于返回的 token 列表 <br>
     * 需要维护实验一所需的符号表条目, 而得在语法分析中才能确定的符号表条目的成员可以先设置为 null
     */
    public void run() {
        int i = 0;
        char[] word = sourceCode.toCharArray();
        State state = State.START;
        String identifier = new String("");
        String number = new String("");
        while (i < word.length) {
            char ch = word[i];
            switch (state) {
                case START:
                    if (Character.isWhitespace(ch)) {
                        state = State.START; // Skip whitespace
                        i++;
                    } else if (Character.isLetter(ch)) {
                        state = State.IN_IDENTIFIER;
                    } else if (Character.isDigit(ch)) {
                        state = State.IN_NUMBER;
                    } else if (isSinglePunctuation(ch)) {
                        state = State.SINGLE_PUNCTUATION;
                    } else {
                        throw new RuntimeException("Unexpected character: " + ch);
                    }
                    break;


                case IN_IDENTIFIER:
                    if (Character.isLetterOrDigit(word[i]) | word[i]=='-') {
                        identifier+=word[i];
                        i++;
                        state = State.IN_IDENTIFIER; // 继续保持在 IN_IDENTIFIER 状态
                    } else {
                        // 一旦遇到非字母字符，处理标识符
                        if (TokenKind.isAllowed(identifier)) {
                            tokens.add(Token.simple(identifier));
                        } else {
                            tokens.add(Token.normal("id", identifier));
                            if (!symbolTable.has(identifier)) {
                                symbolTable.add(identifier);
                            }
                        }
                        state = State.START; // 转回 START 状态
                        identifier="";
                    }
                    break;

                case IN_NUMBER:
                    if (Character.isDigit(word[i])) {
                        number+=word[i];
                        i++;
                        state = State.IN_NUMBER; // 继续保持在 IN_NUMBER 状态
                    } else {
                        // 一旦遇到非数字字符，处理数字字面量
                        tokens.add(Token.normal("IntConst", number));
                        state = State.START; // 转回 START 状态
                        number="";
                    }
                    break;

                case SINGLE_PUNCTUATION:
                    switch (ch) {
                        case '+' -> tokens.add(Token.simple("+"));
                        case '-' -> tokens.add(Token.simple("-"));
                        case '*' -> tokens.add(Token.simple("*"));
                        case '/' -> tokens.add(Token.simple("/"));
                        case '(' -> tokens.add(Token.simple("("));
                        case ')' -> tokens.add(Token.simple(")"));
                        case ',' -> tokens.add(Token.simple(","));
                        case ';' -> tokens.add(Token.simple("Semicolon"));
                        case '=' -> tokens.add(Token.simple("="));
                        default -> throw new RuntimeException("Unexpected symbol: " + ch);
                    }
                    i++;
                    state = State.START;
                    break;
            }
        }

        // 添加EOF标记
        tokens.add(Token.eof());
    }

    private enum State {
        START,
        IN_IDENTIFIER,
        IN_NUMBER,
        SINGLE_PUNCTUATION
    }

    private boolean isSinglePunctuation(char ch) {
        return ",;=+-*/()".indexOf(ch) != -1;
    }

    /**
     * 获得词法分析的结果, 保证在调用了 run 方法之后调用
     *
     * @return Token 列表
     */
    public Iterable<Token> getTokens() {
        // TODO: 从词法分析过程中获取 Token 列表
        // 词法分析过程可以使用 Stream 或 Iterator 实现按需分析
        // 亦可以直接分析完整个文件
        // 总之实现过程能转化为一列表即可
        return tokens;
    }

    public void dumpTokens(String path) {
        FileUtils.writeLines(
            path,
            StreamSupport.stream(getTokens().spliterator(), false).map(Token::toString).toList()
        );
    }


}
